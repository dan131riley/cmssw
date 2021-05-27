#include <stdio.h>
#include <cub/cub.cuh>

#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include "SiStripConditionsGPU.h"
#include "ChanLocsGPU.h"
#include "SiStripRawToClusterGPUKernel.h"
#include "clusterGPU.cuh"

//#define GPU_DEBUG
//#define GPU_CHECK

namespace stripgpu {
  __device__ constexpr int maxseeds() { return MAX_SEEDSTRIPS; }

  __global__ static void setSeedStripsGPU(sst_data_t *sst_data_d, const SiStripConditionsGPU *conditions) {
    const int nStrips = sst_data_d->nStrips;
    const auto __restrict__ chanlocs = sst_data_d->chanlocs;
    const uint8_t *__restrict__ adc = sst_data_d->adc;
    const uint16_t *__restrict__ channels = sst_data_d->channel;
    const uint16_t *__restrict__ stripId = sst_data_d->stripId;
    int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
    int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;
    const float SeedThreshold = sst_data_d->SeedThreshold_;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int i = nthreads * bid + tid;

    if (i < nStrips) {
      seedStripsMask[i] = 0;
      seedStripsNCMask[i] = 0;
      const stripId_t strip = stripId[i];
      if (strip != stripgpu::invStrip) {
        const auto chan = channels[i];
        const fedId_t fed = chanlocs->fedID(chan);
        const fedCh_t channel = chanlocs->fedCh(chan);
        const float noise_i = conditions->noise(fed, channel, strip);
        const uint8_t adc_i = adc[i];
        seedStripsMask[i] = (adc_i >= static_cast<uint8_t>(noise_i * SeedThreshold)) ? 1 : 0;
        seedStripsNCMask[i] = seedStripsMask[i];
      }
    }
  }

  __global__ static void setNCSeedStripsGPU(sst_data_t *sst_data_d, const SiStripConditionsGPU *conditions) {
    const int nStrips = sst_data_d->nStrips;
    const auto __restrict__ chanlocs = sst_data_d->chanlocs;
    const uint16_t *__restrict__ channels = sst_data_d->channel;
    const uint16_t *__restrict__ stripId = sst_data_d->stripId;
    const int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
    int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int i = nthreads * bid + tid;

    if (i > 0 && i < nStrips) {
      const auto detid = chanlocs->detID(channels[i]);
      const auto detid1 = chanlocs->detID(channels[i - 1]);

      if (seedStripsMask[i] && seedStripsMask[i - 1] && (stripId[i] - stripId[i - 1]) == 1 && (detid == detid1))
        seedStripsNCMask[i] = 0;
    }
  }

  __global__ static void setStripIndexGPU(sst_data_t *sst_data_d) {
    const int nStrips = sst_data_d->nStrips;
    const int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;
    const int *__restrict__ prefixSeedStripsNCMask = sst_data_d->prefixSeedStripsNCMask;
    int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int i = nthreads * bid + tid;

    if (i < nStrips) {
      if (seedStripsNCMask[i] == 1) {
        const int index = prefixSeedStripsNCMask[i];
        seedStripsNCIndex[index] = i;
      }
    }
  }

  __global__ static void findLeftRightBoundaryGPU(sst_data_t *sst_data_d,
                                                  const SiStripConditionsGPU *conditions,
                                                  SiStripClustersCUDA::DeviceView *clust_data_d) {
    const int nStrips = sst_data_d->nStrips;
    const int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;
    const auto __restrict__ chanlocs = sst_data_d->chanlocs;
    const uint16_t *__restrict__ stripId = sst_data_d->stripId;
    const uint16_t *__restrict__ channels = sst_data_d->channel;
    const uint8_t *__restrict__ adc = sst_data_d->adc;
    const int nSeedStripsNC = std::min(maxseeds(), *(sst_data_d->prefixSeedStripsNCMask + nStrips - 1));
    const uint8_t MaxSequentialHoles = sst_data_d->MaxSequentialHoles_;
    const float ChannelThreshold = sst_data_d->ChannelThreshold_;
    const float ClusterThresholdSquared = sst_data_d->ClusterThresholdSquared_;
    const int ClusterSizeLimit = sst_data_d->clusterSizeLimit_;

    uint32_t *__restrict__ clusterIndexLeft = clust_data_d->clusterIndex_;
    uint32_t *__restrict__ clusterSize = clust_data_d->clusterSize_;
    detId_t *__restrict__ clusterDetId = clust_data_d->clusterDetId_;
    stripId_t *__restrict__ firstStrip = clust_data_d->firstStrip_;
    bool *__restrict__ trueCluster = clust_data_d->trueCluster_;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;

    const int i = nthreads * bid + tid;

    if (i < nSeedStripsNC) {
      const auto index = seedStripsNCIndex[i];
      const auto chan = channels[index];
      const auto fed = chanlocs->fedID(chan);
      const auto channel = chanlocs->fedCh(chan);
      const auto det = chanlocs->detID(chan);
      const auto strip = stripId[index];
      const auto noise_i = conditions->noise(fed, channel, strip);

      auto noiseSquared_i = noise_i * noise_i;
      float adcSum_i = static_cast<float>(adc[index]);

      // find left boundary
      auto indexLeft = index;
      auto testIndex = index - 1;
      auto size = 1;

      if (testIndex >= 0 && stripId[testIndex] == stripgpu::invStrip) {
        testIndex -= 2;
      }

      if (testIndex >= 0) {
        auto rangeLeft = stripId[indexLeft] - stripId[testIndex] - 1;
        auto testchan = channels[testIndex];
        auto testDet = chanlocs->detID(testchan);
        auto sameDetLeft = det == testDet;

        while (sameDetLeft && rangeLeft >= 0 && rangeLeft <= MaxSequentialHoles && size < ClusterSizeLimit) {
          testchan = channels[testIndex];
          const auto testFed = chanlocs->fedID(testchan);
          const auto testChannel = chanlocs->fedCh(testchan);
          const auto testStrip = stripId[testIndex];
          const auto testNoise = conditions->noise(testFed, testChannel, testStrip);
          const auto testADC = adc[testIndex];

          if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
            ++size;
            indexLeft = testIndex;
            noiseSquared_i += testNoise * testNoise;
            adcSum_i += static_cast<float>(testADC);
          }
          --testIndex;
          if (testIndex >= 0 && stripId[testIndex] == stripgpu::invStrip) {
            testIndex -= 2;
          }
          if (testIndex >= 0) {
            rangeLeft = stripId[indexLeft] - stripId[testIndex] - 1;
            const auto newchan = channels[testIndex];
            const auto newdet = chanlocs->detID(newchan);
            sameDetLeft = det == newdet;
          } else {
            sameDetLeft = false;
          }
        }
      }

      // find right boundary
      auto indexRight = index;
      testIndex = index + 1;

      if (testIndex < nStrips && stripId[testIndex] == stripgpu::invStrip) {
        testIndex += 2;
      }

      if (testIndex < nStrips) {
        auto rangeRight = stripId[testIndex] - stripId[indexRight] - 1;
        auto testchan = channels[testIndex];
        auto testDet = chanlocs->detID(testchan);
        auto sameDetRight = det == testDet;

        while (sameDetRight && rangeRight >= 0 && rangeRight <= MaxSequentialHoles && size < ClusterSizeLimit) {
          testchan = channels[testIndex];
          const auto testFed = chanlocs->fedID(testchan);
          const auto testChannel = chanlocs->fedCh(testchan);
          const auto testStrip = stripId[testIndex];
          const auto testNoise = conditions->noise(testFed, testChannel, testStrip);
          const auto testADC = adc[testIndex];

          if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
            ++size;
            indexRight = testIndex;
            noiseSquared_i += testNoise * testNoise;
            adcSum_i += static_cast<float>(testADC);
          }
          ++testIndex;
          if (testIndex < nStrips && stripId[testIndex] == stripgpu::invStrip) {
            testIndex += 2;
          }
          if (testIndex < nStrips) {
            rangeRight = stripId[testIndex] - stripId[indexRight] - 1;
            const auto newchan = channels[testIndex];
            const auto newdet = chanlocs->detID(newchan);
            sameDetRight = det == newdet;
          } else {
            sameDetRight = false;
          }
        }
      }
      clusterIndexLeft[i] = indexLeft;
      clusterSize[i] = indexRight - indexLeft + 1;
      clusterDetId[i] = det;
      firstStrip[i] = stripId[indexLeft];
      trueCluster[i] =
          (noiseSquared_i * ClusterThresholdSquared <= adcSum_i * adcSum_i) and (clusterSize[i] <= ClusterSizeLimit);
    }
  }

  __global__ static void checkClusterConditionGPU(sst_data_t *sst_data_d,
                                                  const SiStripConditionsGPU *conditions,
                                                  SiStripClustersCUDA::DeviceView *clust_data_d) {
    const uint16_t *__restrict__ stripId = sst_data_d->stripId;
    const auto __restrict__ chanlocs = sst_data_d->chanlocs;
    const uint16_t *__restrict__ channels = sst_data_d->channel;
    const uint8_t *__restrict__ adc = sst_data_d->adc;
    const int nSeedStripsNC = std::min(maxseeds(), *(sst_data_d->prefixSeedStripsNCMask + sst_data_d->nStrips - 1));
    const float minGoodCharge = sst_data_d->minGoodCharge_;  //1620.0;
    const uint32_t *__restrict__ clusterIndexLeft = clust_data_d->clusterIndex_;

    uint32_t *__restrict__ clusterSize = clust_data_d->clusterSize_;
    uint8_t *__restrict__ clusterADCs = clust_data_d->clusterADCs_;
    bool *__restrict__ trueCluster = clust_data_d->trueCluster_;
    float *__restrict__ barycenter = clust_data_d->barycenter_;
    float *__restrict__ charge = clust_data_d->charge_;

    constexpr uint16_t stripIndexMask = 0x7FFF;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int i = nthreads * bid + tid;

    if (i < nSeedStripsNC) {
      if (trueCluster[i]) {
        const int left = clusterIndexLeft[i];
        const int size = clusterSize[i];

        if (i > 0 && clusterIndexLeft[i - 1] == left) {
          trueCluster[i] = 0;  // ignore duplicates
        } else {
          float adcSum = 0.0f;
          int sumx = 0;
          int suma = 0;

          auto j = 0;
          for (int k = 0; k < size; k++) {
            const auto index = left + k;
            const auto chan = channels[index];
            const auto fed = chanlocs->fedID(chan);
            const auto channel = chanlocs->fedCh(chan);
            const auto strip = stripId[index];
#ifdef GPU_CHECK
            if (fed == stripgpu::invFed) {
              printf("Invalid fed index %d\n", index);
            }
#endif
            if (strip != stripgpu::invStrip) {
              const float gain_j = conditions->gain(fed, channel, strip);

              uint8_t adc_j = adc[index];
              const int charge = static_cast<int>(static_cast<float>(adc_j) / gain_j + 0.5f);

              if (adc_j < 254) {
                adc_j = (charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
              }
              if (j < kClusterMaxStrips) {
                clusterADCs[j * nSeedStripsNC + i] = adc_j;
              }
              adcSum += static_cast<float>(adc_j);
              sumx += j * adc_j;
              suma += adc_j;
              j++;
            }
          }
          const auto chan = channels[left];
          const fedId_t fed = chanlocs->fedID(chan);
          const fedCh_t channel = chanlocs->fedCh(chan);
          clusterSize[i] = j;
          charge[i] = adcSum;
          trueCluster[i] = (adcSum * conditions->invthick(fed, channel)) > minGoodCharge;
          barycenter[i] = static_cast<float>(stripId[left] & stripIndexMask) +
                          static_cast<float>(sumx) / static_cast<float>(suma) + 0.5f;
        }
      }
    }
  }

  void SiStripRawToClusterGPUKernel::allocateSSTDataGPU(int max_strips, cudaStream_t stream) {
    stripdata_->seedStripsMask_ = cms::cuda::make_device_unique<int[]>(2 * max_strips, stream);
    stripdata_->prefixSeedStripsNCMask_ = cms::cuda::make_device_unique<int[]>(2 * max_strips, stream);

    sst_data_d_->chanlocs = chanlocsGPU_->chanLocStruct();
    sst_data_d_->stripId = stripdata_->stripIdGPU_.get();
    sst_data_d_->channel = stripdata_->channelGPU_.get();
    sst_data_d_->adc = stripdata_->alldataGPU_.get();
    sst_data_d_->seedStripsMask = stripdata_->seedStripsMask_.get();
    sst_data_d_->prefixSeedStripsNCMask = stripdata_->prefixSeedStripsNCMask_.get();

    sst_data_d_->seedStripsNCMask = sst_data_d_->seedStripsMask + max_strips;
    sst_data_d_->seedStripsNCIndex = sst_data_d_->prefixSeedStripsNCMask + max_strips;

    sst_data_d_->ChannelThreshold_ = ChannelThreshold_;
    sst_data_d_->SeedThreshold_ = SeedThreshold_;
    sst_data_d_->ClusterThresholdSquared_ = ClusterThresholdSquared_;
    sst_data_d_->MaxSequentialHoles_ = MaxSequentialHoles_;
    sst_data_d_->MaxSequentialBad_ = MaxSequentialBad_;
    sst_data_d_->MaxAdjacentBad_ = MaxAdjacentBad_;
    sst_data_d_->minGoodCharge_ = minGoodCharge_;
    sst_data_d_->clusterSizeLimit_ = keepLargeClusters_ ? 256 : kClusterMaxStrips;

    pt_sst_data_d_ = cms::cuda::make_device_unique<sst_data_t>(stream);
    cms::cuda::copyAsync(pt_sst_data_d_, sst_data_d_, stream);
  }

  void SiStripRawToClusterGPUKernel::findClusterGPU(const SiStripConditionsGPU *conditions, cudaStream_t stream) {
    const int nthreads = 128;
    const int nStrips = sst_data_d_->nStrips;
    const int nSeeds = std::min(MAX_SEEDSTRIPS, nStrips);
    const int nblocks = (nSeeds + nthreads - 1) / nthreads;

#ifdef GPU_DEBUG
    auto cpu_index = cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_strip = cms::cuda::make_host_unique<uint16_t[]>(nStrips, stream);
    auto cpu_adc = cms::cuda::make_host_unique<uint8_t[]>(nStrips, stream);
    auto cpu_noise = cms::cuda::make_host_unique<float[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(
        cpu_strip.get(), sst_data_d_->stripId, nStrips * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(
        cudaMemcpyAsync(cpu_adc.get(), sst_data_d_->adc, nStrips * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(
        cpu_index.get(), sst_data_d_->seedStripsNCIndex, nStrips * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    for (int i = 0; i < nStrips; i++) {
      std::cout << " cpu_strip " << cpu_strip[i] << " cpu_adc " << (unsigned int)cpu_adc[i] << " cpu index "
                << cpu_index[i] << std::endl;
    }
#endif

    cudaCheck(cudaMemcpyAsync(&(clusters_d_.nClusters_h),
                              sst_data_d_->prefixSeedStripsNCMask + sst_data_d_->nStrips - 1,
                              sizeof(int),
                              cudaMemcpyDeviceToHost,
                              stream));
    auto clust_data_d = clusters_d_.view();
    findLeftRightBoundaryGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get(), conditions, clust_data_d);
    cudaCheck(cudaGetLastError());
#ifdef GPU_CHECK
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

    checkClusterConditionGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get(), conditions, clust_data_d);
    cudaCheck(cudaGetLastError());

#ifdef GPU_CHECK
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

#ifdef GPU_DEBUG
    cudaStreamSynchronize(stream);
    auto clust_data = clusters_d_.hostView(kClusterMaxStrips, stream);
    cudaStreamSynchronize(stream);

    auto clusterIndexLeft = clust_data->clusterIndex_h.get();
    auto clusterSize = clust_data->clusterSize_h.get();
    auto trueCluster = clust_data->trueCluster_h.get();
    auto ADCs = clust_data->clusterADCs_h.get();
    auto detids = clust_data->clusterDetId_h.get();

    const int nSeedStripsNC = clusters_d_.nClusters_h;
    std::cout << "findClusterGPU nSeedStripsNC=" << nSeedStripsNC << std::endl;

    for (int i = 0; i < nSeedStripsNC; i++) {
      if (trueCluster[i]) {
        int left = clusterIndexLeft[i];
        uint32_t size = clusterSize[i];
        const auto detid = detids[i];
        std::cout << "i=" << i << " detId " << detid << " left " << left << " size " << size << " : ";
        size = std::min(size, kClusterMaxStrips);
        for (uint32_t j = 0; j < size; j++) {
          std::cout << (unsigned int)ADCs[j * nSeedStripsNC + i] << " ";
        }
        std::cout << std::endl;
      }
    }
#endif
  }

  void SiStripRawToClusterGPUKernel::setSeedStripsNCIndexGPU(const SiStripConditionsGPU *conditions,
                                                             cudaStream_t stream) {
#ifdef GPU_DEBUG
    int nStrips = sst_data_d_->nStrips;
    auto cpu_strip = cms::cuda::make_host_unique<uint16_t[]>(nStrips, stream);
    auto cpu_adc = cms::cuda::make_host_unique<uint8_t[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(
        cpu_strip.get(), sst_data_d_->stripId, nStrips * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(
        cudaMemcpyAsync(cpu_adc.get(), sst_data_d_->adc, nStrips * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    for (int i = 0; i < nStrips; i++) {
      std::cout << " cpu_strip " << cpu_strip[i] << " cpu_adc " << (unsigned int)cpu_adc[i] << std::endl;
    }
#endif

    int nthreads = 256;
    int nblocks = (sst_data_d_->nStrips + nthreads - 1) / nthreads;

    //mark seed strips
    setSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get(), conditions);
    cudaCheck(cudaGetLastError());

    //mark only non-consecutive seed strips (mask out consecutive seed strips)
    setNCSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get(), conditions);
    cudaCheck(cudaGetLastError());

    std::size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr,
                                  temp_storage_bytes,
                                  sst_data_d_->seedStripsNCMask,
                                  sst_data_d_->prefixSeedStripsNCMask,
                                  sst_data_d_->nStrips,
                                  stream);
#ifdef GPU_DEBUG
    std::cout << "temp_storage_bytes=" << temp_storage_bytes << std::endl;
#endif

    {
      auto d_temp_storage = cms::cuda::make_device_unique<uint8_t[]>(temp_storage_bytes, stream);
      cub::DeviceScan::ExclusiveSum(d_temp_storage.get(),
                                    temp_storage_bytes,
                                    sst_data_d_->seedStripsNCMask,
                                    sst_data_d_->prefixSeedStripsNCMask,
                                    sst_data_d_->nStrips,
                                    stream);
    }

    setStripIndexGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d_.get());
    cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
    auto cpu_mask = cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_prefix = cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_index = cms::cuda::make_host_unique<int[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(&(sst_data_d_->nSeedStripsNC),
                              sst_data_d_->prefixSeedStripsNCMask + sst_data_d_->nStrips - 1,
                              sizeof(int),
                              cudaMemcpyDeviceToHost,
                              stream));
    cudaCheck(cudaMemcpyAsync(
        cpu_mask.get(), sst_data_d_->seedStripsNCMask, nStrips * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(
        cpu_prefix.get(), sst_data_d_->prefixSeedStripsNCMask, nStrips * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(
        cpu_index.get(), sst_data_d_->seedStripsNCIndex, nStrips * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    const int nSeedStripsNC = std::min(MAX_SEEDSTRIPS, sst_data_d_->nSeedStripsNC);
    std::cout << "nStrips=" << nStrips << " nSeedStripsNC=" << sst_data_d_->nSeedStripsNC << std::endl;
    for (int i = 0; i < nStrips; i++) {
      std::cout << " i " << i << " mask " << cpu_mask[i] << " prefix " << cpu_prefix[i] << " index " << cpu_index[i]
                << std::endl;
    }
#endif
  }
}  // namespace stripgpu
