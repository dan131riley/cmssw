#include <stdio.h>
#include <cub/cub.cuh>

#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"


#include "clusterGPU.cuh"

#define GPU_DEBUG

using fedId_t = stripgpu::fedId_t;
using fedCh_t = stripgpu::fedCh_t;
using detId_t = stripgpu::detId_t;
using stripId_t = stripgpu::stripId_t;

__global__
static void setSeedStripsGPU(sst_data_t *sst_data_d, const SiStripConditionsGPU * conditions) {
  const int nStrips = sst_data_d->nStrips;
  const uint8_t *__restrict__ adc = sst_data_d->adc;
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  const fedId_t *__restrict__ fedId = sst_data_d->fedId;
  const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;
  int *__restrict__ seedStripsMask = sst_data_d->seedStripsMask;
  int *__restrict__ seedStripsNCMask = sst_data_d->seedStripsNCMask;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const float SeedThreshold = 3.0;

  int i = nthreads * bid + tid;

  if (i<nStrips) {
    seedStripsMask[i] = 0;
    seedStripsNCMask[i] = 0;
    stripId_t strip = stripId[i];
    if (strip != stripgpu::invStrip) {
      fedId_t fed = fedId[i];
      fedCh_t channel = fedCh[i];
      float noise_i = conditions->noise(fed, channel, strip);
      uint8_t adc_i = adc[i];
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

  int i = nthreads * bid + tid;

  if (i>0&&i<nStrips) {
    if (seedStripsMask[i]&&seedStripsMask[i-1]&&(stripId[i]-stripId[i-1])==1&&(detId[i]==detId[i-1])) seedStripsNCMask[i] = 0;
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

  int i = nthreads * bid + tid;

  if (i<nStrips) {
    if (seedStripsNCMask[i] == 1) {
      int index = prefixSeedStripsNCMask[i];
      seedStripsNCIndex[index] = i;
    }
  }
}

__global__
static void findLeftRightBoundaryGPU(sst_data_t *sst_data_d, const SiStripConditionsGPU *conditions, clust_data_t *clust_data_d) {
  const int nStrips = sst_data_d->nStrips;
  const int *__restrict__ seedStripsNCIndex = sst_data_d->seedStripsNCIndex;
  const int nSeedStripsNC = sst_data_d->nSeedStripsNC;
  const uint16_t *__restrict__ stripId = sst_data_d->stripId;
  const detId_t *__restrict__ detId = sst_data_d->detId;
  const uint8_t *__restrict__ adc = sst_data_d->adc;
  const fedId_t *__restrict__ fedId = sst_data_d->fedId;
  const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;

  int *__restrict__ clusterLastIndexLeft = clust_data_d->clusterLastIndexLeft;
  int *__restrict__ clusterLastIndexRight = clust_data_d->clusterLastIndexRight;
  bool *__restrict__ trueCluster = clust_data_d->trueCluster;

   const uint8_t MaxSequentialHoles = 0;
   const float  ChannelThreshold = 2.0;
   const float ClusterThresholdSquared = 25.0;

   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   int index, testIndexLeft, testIndexRight, indexLeft, indexRight, rangeLeft, rangeRight;
   uint8_t testADC;
   float noise_i, testNoise, noiseSquared_i, adcSum_i;
   bool noiseSquaredPass, sameDetLeft, sameDetRight;
   fedId_t fed, testFed;
   fedCh_t channel, testChannel;
   stripId_t strip, testStrip;
   int i = nthreads * bid + tid;

   if (i<nSeedStripsNC) {
     index=seedStripsNCIndex[i];
     indexLeft = index;
     indexRight = index;
     fed = fedId[index];
     channel = fedCh[index];
     strip = stripId[index];
     noise_i = conditions->noise(fed, channel, strip);
     noiseSquared_i = noise_i*noise_i;
     adcSum_i = static_cast<float>(adc[index]);

     // find left boundary
     testIndexLeft=index-1;
     if (testIndexLeft>=0) {
       rangeLeft = stripId[indexLeft]-stripId[testIndexLeft]-1;
       sameDetLeft = detId[index] == detId[testIndexLeft];
       while(sameDetLeft&&testIndexLeft>=0&&rangeLeft>=0&&rangeLeft<=MaxSequentialHoles) {
         testFed = fedId[testIndexLeft];
         testChannel = fedCh[testIndexLeft];
         testStrip = stripId[testIndexLeft];
         testNoise = conditions->noise(testFed, testChannel, testStrip);
         testADC = adc[testIndexLeft];

         if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
           --indexLeft;
           noiseSquared_i += testNoise*testNoise;
           adcSum_i += static_cast<float>(testADC);
         }
         --testIndexLeft;
         if (testIndexLeft>=0) {
           rangeLeft = stripId[indexLeft]-stripId[testIndexLeft]-1;
           sameDetLeft = detId[index] == detId[testIndexLeft];
         }
       }
     }

     // find right boundary
     testIndexRight=index+1;
     if (testIndexRight<nStrips) {
       rangeRight = stripId[testIndexRight]-stripId[indexRight]-1;
       sameDetRight = detId[index] == detId[testIndexRight];
       while(sameDetRight&&testIndexRight<nStrips&&rangeRight>=0&&rangeRight<=MaxSequentialHoles) {
         testFed = fedId[testIndexRight];
         testChannel = fedCh[testIndexRight];
         testStrip = stripId[testIndexRight];
         testNoise = conditions->noise(testFed, testChannel, testStrip);
         testADC = adc[testIndexRight];
         if (testADC >= static_cast<uint8_t>(testNoise * ChannelThreshold)) {
           ++indexRight;
           noiseSquared_i += testNoise*testNoise;
           adcSum_i += static_cast<float>(testADC);
         }
         ++testIndexRight;
         if (testIndexRight<nStrips) {
           rangeRight = stripId[testIndexRight]-stripId[indexRight]-1;
           sameDetRight = detId[index] == detId[testIndexRight];
         }
       }
     }
     noiseSquaredPass = noiseSquared_i*ClusterThresholdSquared <= adcSum_i*adcSum_i;
     trueCluster[i] = noiseSquaredPass;
     clusterLastIndexLeft[i] = indexLeft;
     clusterLastIndexRight[i] = indexRight;

   }
}

__global__
static void checkClusterConditionGPU(sst_data_t *sst_data_d, const SiStripConditionsGPU *conditions, clust_data_t *clust_data_d) {
   const uint16_t *__restrict__ stripId = sst_data_d->stripId;
   const uint8_t *__restrict__ adc = sst_data_d->adc;
   const fedId_t *__restrict__ fedId = sst_data_d->fedId;
   const fedCh_t *__restrict__ fedCh = sst_data_d->fedCh;
   const int nSeedStripsNC = sst_data_d->nSeedStripsNC;
   const int *__restrict__ clusterLastIndexLeft = clust_data_d->clusterLastIndexLeft;
   const int *__restrict__ clusterLastIndexRight = clust_data_d->clusterLastIndexRight;
   uint8_t *__restrict__ clusterADCs = clust_data_d->clusterADCs;
   bool *__restrict__ trueCluster = clust_data_d->trueCluster;
   float *__restrict__ barycenter = clust_data_d->barycenter;
   const float minGoodCharge = 1620.0;
   const uint16_t stripIndexMask = 0x7FFF;

   const int tid = threadIdx.x;
   const int bid = blockIdx.x;
   const int nthreads = blockDim.x;

   const int i = nthreads * bid + tid;

   int left, right, size, j;
   int charge;
   uint8_t adc_j;
   float gain_j;
   float adcSum=0.0f;
   int sumx=0;
   int suma=0;
   fedId_t fed;
   fedCh_t channel;
   stripId_t strip;

   if (i<nSeedStripsNC) {
     if (trueCluster[i]) {
       left=clusterLastIndexLeft[i];
       right=clusterLastIndexRight[i];
       size=right-left+1;

       if (i>0 && clusterLastIndexLeft[i-1]==left) {
         trueCluster[i] = 0;  // ignore duplicates
       } else {
         for (j=0; j<size; j++){
           adc_j = adc[left+j];
           fed = fedId[left+j];
           channel = fedCh[left+j];
           strip = stripId[left+j];
           gain_j = conditions->gain(fed, channel, strip);
           charge = static_cast<int>( static_cast<float>(adc_j)/gain_j + 0.5f );
           if (adc_j < 254) adc_j = ( charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
           if (j < kClusterMaxStrips) {
             clusterADCs[j*nSeedStripsNC+i] = adc_j;
           }
           adcSum += static_cast<float>(adc_j);
           sumx += j*adc_j;
           suma += adc_j;
         }
         barycenter[i] = static_cast<float>(stripId[left] & stripIndexMask) + static_cast<float>(sumx)/static_cast<float>(suma) + 0.5f;
       }
       trueCluster[i] = (adcSum*conditions->invthick(fed, channel)) > minGoodCharge;
     }
   }
}

void allocateSSTDataGPU(int max_strips, StripDataGPU& stripdata, sst_data_t *sst_data_d, sst_data_t **pt_sst_data_d, cudaStream_t stream) {
  int dev = cms::cuda::currentDevice();
  *pt_sst_data_d = (sst_data_t *)cms::cuda::allocate_device(dev, sizeof(sst_data_t), stream);
  sst_data_d->detId = stripdata.detIdGPU_.get();
  sst_data_d->stripId = stripdata.stripIdGPU_.get();
  sst_data_d->adc = stripdata.alldataGPU_.get();
  sst_data_d->fedId = stripdata.fedIdGPU_.get();
  sst_data_d->fedCh = stripdata.fedChGPU_.get();
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
  cudaCheck(cudaMemcpyAsync((void *)*pt_sst_data_d, sst_data_d, sizeof(sst_data_t), cudaMemcpyHostToDevice, stream));
}

void allocateClustDataGPU(int max_strips, clust_data_t *clust_data_d, clust_data_t **pt_clust_data_d, cudaStream_t stream) {
  int dev = cms::cuda::currentDevice();

  *pt_clust_data_d = (clust_data_t *)cms::cuda::allocate_device(dev, sizeof(clust_data_t), stream);
  clust_data_d->clusterLastIndexLeft = (int *)cms::cuda::allocate_device(dev, 2*max_strips*sizeof(int), stream);
  clust_data_d->clusterADCs = (uint8_t *)cms::cuda::allocate_device(dev, max_strips*kClusterMaxStrips*sizeof(uint8_t), stream);
  clust_data_d->trueCluster = (bool *)cms::cuda::allocate_device(dev, max_strips*sizeof(bool), stream);
  clust_data_d->barycenter = (float *)cms::cuda::allocate_device(dev, max_strips*sizeof(float), stream);
  clust_data_d->clusterLastIndexRight = clust_data_d->clusterLastIndexLeft + max_strips;
  cudaCheck(cudaMemcpyAsync((void *)*pt_clust_data_d, clust_data_d, sizeof(clust_data_t), cudaMemcpyHostToDevice, stream));
}

void allocateClustData(int max_seedstrips, clust_data_t *clust_data, cudaStream_t stream){
  clust_data->clusterLastIndexLeft = (int *)cms::cuda::allocate_host(2*max_seedstrips*sizeof(int), stream);
  clust_data->clusterLastIndexRight = clust_data->clusterLastIndexLeft + max_seedstrips;
  clust_data->clusterADCs = (uint8_t*)cms::cuda::allocate_host(max_seedstrips*kClusterMaxStrips*sizeof(uint8_t), stream);
  clust_data->trueCluster = (bool *)cms::cuda::allocate_host(max_seedstrips*sizeof(bool), stream);
  clust_data->barycenter = (float *)cms::cuda::allocate_host(max_seedstrips*sizeof(float), stream);
}

void freeSSTDataGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, cudaStream_t stream) {
  int dev = cms::cuda::currentDevice();
  cms::cuda::free_device(dev, pt_sst_data_d);
  cms::cuda::free_device(dev, sst_data_d->seedStripsMask);
  cms::cuda::free_device(dev, sst_data_d->prefixSeedStripsNCMask);
}

void freeClustDataGPU(clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, cudaStream_t stream) {
  int dev = cms::cuda::currentDevice();
  cms::cuda::free_device(dev, pt_clust_data_d);
  cms::cuda::free_device(dev, clust_data_d->clusterLastIndexLeft);
  cms::cuda::free_device(dev, clust_data_d->clusterADCs);
  cms::cuda::free_device(dev, clust_data_d->trueCluster);
  cms::cuda::free_device(dev, clust_data_d->barycenter);
}

void freeClustData(clust_data_t *clust_data) {
  cms::cuda::free_host(clust_data->clusterLastIndexLeft);
  cms::cuda::free_host(clust_data->clusterADCs);
  cms::cuda::free_host(clust_data->trueCluster);
  cms::cuda::free_host(clust_data->barycenter);
}

void findClusterGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, const SiStripConditionsGPU *conditions, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, cudaStream_t stream) {
  const int nthreads = 128;
  const int nSeedStripsNC = sst_data_d->nSeedStripsNC;
  const int nblocks = (nSeedStripsNC+nthreads-1)/nthreads;

#ifdef GPU_DEBUG
  int nStrips = sst_data_d->nStrips;
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

  std::cout << "findLeftRightBoundaryGPU sync"; cudaStreamSynchronize(stream);
  findLeftRightBoundaryGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d, conditions, pt_clust_data_d);
  cudaCheck(cudaGetLastError());

  std::cout << "checkClusterConditionGPU sync"; cudaStreamSynchronize(stream);
  checkClusterConditionGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d, conditions, pt_clust_data_d);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  auto clusterLastIndexLeft = cms::cuda::make_host_unique<int[]>(nSeedStripsNC, stream);
  auto clusterLastIndexRight = cms::cuda::make_host_unique<int[]>(nSeedStripsNC, stream);
  auto trueCluster = cms::cuda::make_host_unique<bool[]>(nSeedStripsNC, stream);
  auto ADCs = cms::cuda::make_host_unique<uint8_t[]>(nSeedStripsNC*kClusterMaxStrips, stream);

  std::cout<< "findClusterGPU sync"; cudaStreamSynchronize(stream);

  std::cout<<"findClusterGPU nSeedStripsNC="<<nSeedStripsNC<<std::endl;
  cudaCheck(cudaMemcpyAsync(clusterLastIndexLeft.get(), clust_data_d->clusterLastIndexLeft, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost, 
stream));
  cudaCheck(cudaMemcpyAsync(clusterLastIndexRight.get(), clust_data_d->clusterLastIndexRight, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync(trueCluster.get(), clust_data_d->trueCluster, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync(ADCs.get(), clust_data_d->clusterADCs, nSeedStripsNC*kClusterMaxStrips*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));

  cudaStreamSynchronize(stream);

  for (int i=0; i<nSeedStripsNC; i++) {
    if (trueCluster[i]){
      int left=clusterLastIndexLeft[i];
      int right=clusterLastIndexRight[i];
      std::cout<<"i="<<i<<" left "<<left<<" right "<<right<<" : ";
      int size=std::min(right-left+1, kClusterMaxStrips);
      for (int j=0; j<size; j++){
        std::cout<<(unsigned int)ADCs[j*nSeedStripsNC+i]<<" ";
      }
      std::cout<<std::endl;
    }
  }

#endif

}

void setSeedStripsNCIndexGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, const SiStripConditionsGPU *conditions, cudaStream_t stream) {
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

  cudaCheck(cudaMemcpyAsync((void *)&(sst_data_d->nSeedStripsNC), sst_data_d->prefixSeedStripsNCMask+sst_data_d->nStrips-1, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  if (sst_data_d->nSeedStripsNC > MAX_SEEDSTRIPS) {
    printf("Seed strips %d greater than max\n", sst_data_d->nSeedStripsNC);
    sst_data_d->nSeedStripsNC = MAX_SEEDSTRIPS;
  }
  cudaCheck(cudaMemcpyAsync((void *)&(pt_sst_data_d->nSeedStripsNC), &(sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyHostToDevice, stream));

  setStripIndexGPU<<<nblocks, nthreads, 0, stream>>>(pt_sst_data_d);
  cudaCheck(cudaGetLastError());

#ifdef GPU_DEBUG
  auto cpu_mask  = cms::cuda::make_host_unique<int[]>(nStrips, stream);
  auto cpu_prefix= cms::cuda::make_host_unique<int[]>(nStrips, stream);
  auto cpu_index = cms::cuda::make_host_unique<int[]>(nStrips, stream);

  cudaCheck(cudaMemcpyAsync(cpu_mask.get(), sst_data_d->seedStripsNCMask, nStrips*sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync(cpu_prefix.get(), sst_data_d->prefixSeedStripsNCMask, nStrips*sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync(cpu_index.get(), sst_data_d->seedStripsNCIndex, nStrips*sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync(&(sst_data_d->nSeedStripsNC), &(pt_sst_data_d->nSeedStripsNC), sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaStreamSynchronize(stream));

  std::cout<<"nStrips="<<nStrips<<" nSeedStripsNC="<<sst_data_d->nSeedStripsNC<<" temp_storage_bytes="<<sst_data_d->temp_storage_bytes<<std::endl;
  for (int i=0; i<nStrips; i++) {
    std::cout<<" i "<<i<<" mask "<<cpu_mask[i]<<" prefix "<<cpu_prefix[i]<<" index "<<cpu_index[i]<<std::endl;
  }
#endif
}


void cpyGPUToCPU(sst_data_t * sst_data_d, sst_data_t *pt_sst_data_d, clust_data_t *clust_data, clust_data_t *clust_data_d, cudaStream_t stream) {
  const int nSeedStripsNC = sst_data_d->nSeedStripsNC;

  cudaCheck(cudaMemcpyAsync((void *)(clust_data->clusterLastIndexLeft), clust_data_d->clusterLastIndexLeft, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync((void *)(clust_data->clusterLastIndexRight), clust_data_d->clusterLastIndexRight, nSeedStripsNC*sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync((void *)(clust_data->clusterADCs), clust_data_d->clusterADCs, nSeedStripsNC*kClusterMaxStrips*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync((void *)(clust_data->trueCluster), clust_data_d->trueCluster, nSeedStripsNC*sizeof(bool), cudaMemcpyDeviceToHost, stream));
  cudaCheck(cudaMemcpyAsync((void *)(clust_data->barycenter), clust_data_d->barycenter, nSeedStripsNC*sizeof(float), cudaMemcpyDeviceToHost, stream));
}
