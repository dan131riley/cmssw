#include <stdio.h>
#include <cub/cub.cuh>

#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "ChanLocsGPU.h"
#include "SiStripRawToClusterGPUKernel.h"
#include "clusterGPU.cuh"

//#define GPU_DEBUG
//#define GPU_CHECK

namespace stripgpu {
  __device__ constexpr int maxseeds() { return MAX_SEEDSTRIPS; }

  __global__
  static void setSeedStripsGPU(sst_data_t *sst_data_d, const SiStripConditionsGPU *conditions) {
    const int nStrips = sst_data_d->nStrips;
    const uint8_t *__restrict__ adc = sst_data_d->adc;
    const uint16_t *__restrict__ stripId = sst_data_d->stripId;
    const fedId_t *__restrict__ fedId = sst_data_d->fedId;
    const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;
    int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
    int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;

    constexpr float SeedThreshold = 3.0;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int i = nthreads * bid + tid;

    if (i < nStrips) {
      seedStripsMask[i] = 0;
      seedStripsNCMask[i] = 0;
      const stripId_t strip = stripId[i];
      if (strip != stripgpu::invStrip) {
        const fedId_t fed = fedId[i];
        const fedCh_t channel = fedCh[i];
        const float noise_i = conditions->noise(fed, channel, strip);
        const uint8_t adc_i = adc[i];
        seedStripsMask[i] = (adc_i >= static_cast<uint8_t>( noise_i * SeedThreshold)) ? 1:0;
        seedStripsNCMask[i] = seedStripsMask[i];
      }
    }
  }

  __global__
  static void setNCSeedStripsGPU(sst_data_t *sst_data_d) {
    const int nStrips = sst_data_d->nStrips;
    const detId_t *__restrict__ detId = sst_data_d->detId;
    const uint16_t *__restrict__ stripId = sst_data_d->stripId;
    const int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
    int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int i = nthreads * bid + tid;

    if (i > 0 && i < nStrips) {
      if (seedStripsMask[i] && seedStripsMask[i-1] && (stripId[i]-stripId[i-1])==1 && (detId[i]==detId[i-1])) seedStripsNCMask[i] = 0;
    }
  }

  __global__
  static void setStripIndexGPU(sst_data_t *sst_data_d) {
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

  __global__
  static void findLeftRightBoundaryGPU(sst_data_t *sst_data_d, const SiStripConditionsGPU *conditions,
                                       SiStripClustersCUDA::DeviceView *clust_data_d) {
    const int nStrips = sst_data_d->nStrips;
    const int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;
    const uint16_t *__restrict__ stripId = sst_data_d->stripId;
    const detId_t *__restrict__ detId = sst_data_d->detId;
    const uint8_t *__restrict__ adc = sst_data_d->adc;
    const fedId_t *__restrict__ fedId = sst_data_d->fedId;
    const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;
    const int nSeedStripsNC = std::min(maxseeds(), *(sst_data_d->prefixSeedStripsNCMask+nStrips-1));

    uint32_t *__restrict__ clusterIndexLeft = clust_data_d->clusterIndex_;
    uint32_t *__restrict__ clusterSize = clust_data_d->clusterSize_;
    detId_t *__restrict__ clusterDetId = clust_data_d->clusterDetId_;
    stripId_t *__restrict__ firstStrip = clust_data_d->firstStrip_;
    bool *__restrict__ trueCluster = clust_data_d->trueCluster_;

    constexpr uint8_t MaxSequentialHoles = 0;
    constexpr float  ChannelThreshold = 2.0f;
    constexpr float ClusterThresholdSquared = 25.0f;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;

    const int i = nthreads * bid + tid;

    if (i < nSeedStripsNC) {
      const auto index = seedStripsNCIndex[i];
      const auto fed = fedId[index];
      const auto det = detId[index];
      const auto channel = fedCh[index];
      const auto strip = stripId[index];
      const auto noise_i = conditions->noise(fed, channel, strip);

      auto noiseSquared_i = noise_i*noise_i;
      float adcSum_i = static_cast<float>(adc[index]);

      // find left boundary
      auto indexLeft = index;
      auto testIndex = index-1;

      if (testIndex >= 0 && stripId[testIndex] == stripgpu::invStrip) {
        testIndex -= 2;
      }

      if (testIndex >= 0) {
        auto rangeLeft = stripId[indexLeft]-stripId[testIndex]-1;
        auto sameDetLeft = det == detId[testIndex];

        while (sameDetLeft && rangeLeft>=0 && rangeLeft<=MaxSequentialHoles) {
          const auto testFed = fedId[testIndex];
          const auto testChannel = fedCh[testIndex];
          const auto testStrip = stripId[testIndex];
          const auto testNoise = conditions->noise(testFed, testChannel, testStrip);
          const auto testADC = adc[testIndex];

          if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
            indexLeft = testIndex;
            noiseSquared_i += testNoise*testNoise;
            adcSum_i += static_cast<float>(testADC);
          }
          --testIndex;
          if (testIndex >= 0 && stripId[testIndex] == stripgpu::invStrip) {
            testIndex -= 2;
          }
          if (testIndex >= 0) {
            rangeLeft = stripId[indexLeft]-stripId[testIndex]-1;
            sameDetLeft = det == detId[testIndex];
          } else {
            sameDetLeft = false;
          }
        }
      }

      // find right boundary
      auto indexRight = index;
      testIndex = index+1;

      if (testIndex < nStrips && stripId[testIndex] == stripgpu::invStrip) {
        testIndex += 2;
      }
 
      if (testIndex < nStrips) {
        auto rangeRight = stripId[testIndex]-stripId[indexRight]-1;
        auto sameDetRight = det == detId[testIndex];

        while(sameDetRight && rangeRight>=0 && rangeRight<=MaxSequentialHoles) {
          const auto testFed = fedId[testIndex];
          const auto testChannel = fedCh[testIndex];
          const auto testStrip = stripId[testIndex];
          const auto testNoise = conditions->noise(testFed, testChannel, testStrip);
          const auto testADC = adc[testIndex];

          if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
            indexRight = testIndex;
            noiseSquared_i += testNoise*testNoise;
            adcSum_i += static_cast<float>(testADC);
          }
          ++testIndex;
          if (testIndex < nStrips && stripId[testIndex] == stripgpu::invStrip) {
            testIndex += 2;
          }
          if (testIndex < nStrips) {
            rangeRight = stripId[testIndex]-stripId[indexRight]-1;
            sameDetRight = det == detId[testIndex];
          } else {
            sameDetRight = false;
          }
        }
      }
      trueCluster[i] = noiseSquared_i*ClusterThresholdSquared <= adcSum_i*adcSum_i;
      clusterIndexLeft[i] = indexLeft;
      clusterSize[i] = indexRight - indexLeft + 1;
      clusterDetId[i] = det;
      firstStrip[i] = stripId[indexLeft];
    }
  }

  __global__
  static void checkClusterConditionGPU(sst_data_t *sst_data_d, const SiStripConditionsGPU *conditions,
                                       SiStripClustersCUDA::DeviceView *clust_data_d) {
    const uint16_t *__restrict__ stripId = sst_data_d->stripId;
    const uint8_t *__restrict__ adc = sst_data_d->adc;
    const fedId_t *__restrict__ fedId = sst_data_d->fedId;
    const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;
    const uint32_t *__restrict__ clusterIndexLeft = clust_data_d->clusterIndex_;
    const int nSeedStripsNC = std::min(maxseeds(), *(sst_data_d->prefixSeedStripsNCMask+sst_data_d->nStrips-1));

    uint32_t *__restrict__ clusterSize = clust_data_d->clusterSize_;
    uint8_t *__restrict__ clusterADCs = clust_data_d->clusterADCs_;
    bool *__restrict__ trueCluster = clust_data_d->trueCluster_;
    float *__restrict__ barycenter = clust_data_d->barycenter_;

    constexpr float minGoodCharge = -1.0f; //1620.0;
    constexpr uint16_t stripIndexMask = 0x7FFF;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int i = nthreads * bid + tid;

    if (i < nSeedStripsNC) {
      if (trueCluster[i]) {
        const int left = clusterIndexLeft[i];
        const int size = clusterSize[i];

        if (i>0 && clusterIndexLeft[i-1]==left) {
          trueCluster[i] = 0;  // ignore duplicates
        } else {
          float adcSum = 0.0f;
          int sumx = 0;
          int suma = 0;

          auto j = 0;
          for (int k=0; k<size; k++) {
            const auto index = left+k;
            const auto fed = fedId[index];
            const auto channel = fedCh[index];
            const auto strip = stripId[index];
#ifdef GPU_CHECK
            if (fed == stripgpu::invFed) {
              printf("Invalid fed index %d\n", index);
            }
#endif
            if (strip != stripgpu::invStrip) {
              const float gain_j = conditions->gain(fed, channel, strip);
 
              uint8_t adc_j = adc[index];
              const int charge = static_cast<int>( static_cast<float>(adc_j)/gain_j + 0.5f );
 
              if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
              if (j < kClusterMaxStrips) {
                clusterADCs[j*nSeedStripsNC+i] = adc_j;
              }
              adcSum += static_cast<float>(adc_j);
              sumx += j*adc_j;
              suma += adc_j;
              j++;
            }
          }
          const fedId_t fed = fedId[left];
          const fedCh_t channel = fedCh[left];
          trueCluster[i] = (adcSum*conditions->invthick(fed, channel)) > minGoodCharge;
          barycenter[i] = static_cast<float>(stripId[left] & stripIndexMask) + static_cast<float>(sumx)/static_cast<float>(suma) + 0.5f;
          clusterSize[i] = j;
        }
      }
    }
  }

  void SiStripRawToClusterGPUKernel::allocateSSTDataGPU(int max_strips, cudaStream_t stream) {
    int dev = cms::cuda::currentDevice();
    pt_sst_data_d = (sst_data_t *)cms::cuda::allocate_device(dev, sizeof(sst_data_t), stream);
    sst_data_d->detId = stripdata->detIdGPU_.get();
    sst_data_d->stripId = stripdata->stripIdGPU_.get();
    sst_data_d->adc = stripdata->alldataGPU_.get();
    sst_data_d->fedId = stripdata->fedIdGPU_.get();
    sst_data_d->fedCh = stripdata->fedChGPU_.get();
    sst_data_d->seedStripsMask = (int *)cms::cuda::allocate_device(dev, 2*max_strips*sizeof(int), stream);
    sst_data_d->prefixSeedStripsNCMask = (int *)cms::cuda::allocate_device(dev, 2*max_strips*sizeof(int), stream);

    sst_data_d->seedStripsNCMask = sst_data_d->seedStripsMask + max_strips;
    sst_data_d->seedStripsNCIndex = sst_data_d->prefixSeedStripsNCMask + max_strips;
    sst_data_d->d_temp_storage=NULL;
    sst_data_d->temp_storage_bytes=0;
    cub::DeviceScan::ExclusiveSum(sst_data_d->d_temp_storage, sst_data_d->temp_storage_bytes, sst_data_d->seedStripsNCMask, sst_data_d->prefixSeedStripsNCMask, sst_data_d->nStrips);
#ifdef GPU_DEBUG
    std::cout<<"temp_storage_bytes="<<sst_data_d->temp_storage_bytes<<std::endl;
#endif

    sst_data_d->d_temp_storage = cms::cuda::allocate_device(dev, sst_data_d->temp_storage_bytes, stream);
    cudaCheck(cudaMemcpyAsync(pt_sst_data_d, sst_data_d.get(), sizeof(sst_data_t), cudaMemcpyHostToDevice, stream));
  }

  void SiStripRawToClusterGPUKernel::freeSSTDataGPU(cudaStream_t stream) {
    int dev = cms::cuda::currentDevice();
    cms::cuda::free_device(dev, pt_sst_data_d);
    cms::cuda::free_device(dev, sst_data_d->seedStripsMask);
    cms::cuda::free_device(dev, sst_data_d->prefixSeedStripsNCMask);
  }

  void SiStripRawToClusterGPUKernel::findClusterGPU(const SiStripConditionsGPU *conditions, cudaStream_t stream) {
    const int nthreads = 128;
    const int nStrips = sst_data_d->nStrips;
    const int nSeeds = std::min(MAX_SEEDSTRIPS, nStrips);
    const int nblocks = (nSeeds+nthreads-1)/nthreads;

#ifdef GPU_DEBUG
    auto cpu_index = cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_strip = cms::cuda::make_host_unique<uint16_t[]>(nStrips, stream);
    auto cpu_adc = cms::cuda::make_host_unique<uint8_t[]>(nStrips, stream);
    auto cpu_noise = cms::cuda::make_host_unique<float[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(cpu_strip.get(), sst_data_d->stripId, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(cpu_adc.get(), sst_data_d->adc, nStrips*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(cpu_index.get(), sst_data_d->seedStripsNCIndex, nStrips*sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    for (int i=0; i<nStrips; i++) {
      std::cout<<" cpu_strip "<<cpu_strip[i]<<" cpu_adc "<<(unsigned int)cpu_adc[i]<<" cpu index "<<cpu_index[i]<<std::endl;
    }
#endif

    cudaCheck(cudaMemcpyAsync(&(clusters_d.nClusters_h), sst_data_d->prefixSeedStripsNCMask+sst_data_d->nStrips-1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    auto clust_data_d = clusters_d.view();
    findLeftRightBoundaryGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d, conditions, clust_data_d);
    cudaCheck(cudaGetLastError());
#ifdef GPU_CHECK
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

    checkClusterConditionGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d, conditions, clust_data_d);
    cudaCheck(cudaGetLastError());

#ifdef GPU_CHECK
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError());
#endif

#ifdef GPU_DEBUG
    cudaCheck(cudaMemcpyAsync(&(sst_data_d->nSeedStripsNC), sst_data_d->prefixSeedStripsNCMask+sst_data_d->nStrips-1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    auto clusterIndexLeft = clust_data->clusterIndexLeft;
    auto clusterSize = clust_data->clusterSize;
    auto trueCluster = clust_data->trueCluster;
    auto ADCs = clust_data->clusterADCs;
    auto detIds = clust_data->clusterDetId;
    //auto stripIds = clust_data->firstStrip;

    const int nSeedStripsNC = std::min(MAX_SEEDSTRIPS, sst_data_d->nSeedStripsNC);
    std::cout<<"findClusterGPU nSeedStripsNC="<<nSeedStripsNC<<std::endl;

    for (int i=0; i<nSeedStripsNC; i++) {
      if (trueCluster[i]){
        int left=clusterIndexLeft[i];
        int size=clusterSize[i];
        auto detid = detIds[i];
        std::cout<<"i="<<i<<" detId "<<detid<<" left "<<left<<" size "<<size<<" : ";
        size=std::min(size, kClusterMaxStrips);
        for (int j=0; j<size; j++){
          std::cout<<(unsigned int)ADCs[j*nSeedStripsNC+i]<<" ";
        }
        std::cout<<std::endl;
      }
    }
#endif
  }

  void SiStripRawToClusterGPUKernel::setSeedStripsNCIndexGPU(const SiStripConditionsGPU *conditions, cudaStream_t stream) {
#ifdef GPU_DEBUG
    int nStrips = sst_data_d->nStrips;
    auto cpu_strip = cms::cuda::make_host_unique<uint16_t[]>(nStrips, stream);
    auto cpu_adc = cms::cuda::make_host_unique<uint8_t[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(cpu_strip.get(), sst_data_d->stripId, nStrips*sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(cpu_adc.get(), sst_data_d->adc, nStrips*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    for (int i=0; i<nStrips; i++) {
      std::cout<<" cpu_strip "<<cpu_strip[i]<<" cpu_adc "<<(unsigned int)cpu_adc[i]<<std::endl;
    }
#endif

    int nthreads = 256;
    int nblocks = (sst_data_d->nStrips+nthreads-1)/nthreads;

    //mark seed strips
    setSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d, conditions);
    cudaCheck(cudaGetLastError());

    //mark only non-consecutive seed strips (mask out consecutive seed strips)
    setNCSeedStripsGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d);
    cudaCheck(cudaGetLastError());

    cub::DeviceScan::ExclusiveSum(sst_data_d->d_temp_storage, sst_data_d->temp_storage_bytes, sst_data_d->seedStripsNCMask, sst_data_d->prefixSeedStripsNCMask, sst_data_d->nStrips, stream);

    setStripIndexGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d);
    cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
    auto cpu_mask  = cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_prefix= cms::cuda::make_host_unique<int[]>(nStrips, stream);
    auto cpu_index = cms::cuda::make_host_unique<int[]>(nStrips, stream);

    cudaCheck(cudaMemcpyAsync(&(sst_data_d->nSeedStripsNC), sst_data_d->prefixSeedStripsNCMask+sst_data_d->nStrips-1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(cpu_mask.get(), sst_data_d->seedStripsNCMask, nStrips*sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(cpu_prefix.get(), sst_data_d->prefixSeedStripsNCMask, nStrips*sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaMemcpyAsync(cpu_index.get(), sst_data_d->seedStripsNCIndex, nStrips*sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    const int nSeedStripsNC = std::min(MAX_SEEDSTRIPS, sst_data_d->nSeedStripsNC);
    std::cout<<"nStrips="<<nStrips<<" nSeedStripsNC="<<sst_data_d->nSeedStripsNC<<" temp_storage_bytes="<<sst_data_d->temp_storage_bytes<<std::endl;
    for (int i=0; i<nStrips; i++) {
      std::cout<<" i "<<i<<" mask "<<cpu_mask[i]<<" prefix "<<cpu_prefix[i]<<" index "<<cpu_index[i]<<std::endl;
    }
#endif
  }
}
