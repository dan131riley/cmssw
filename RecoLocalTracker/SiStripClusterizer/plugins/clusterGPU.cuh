#ifndef _CLUSTER_GPU_KERNEL_
#define _CLUSTER_GPU_KERNEL_

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripConditionsGPU.h"
#include "unpackGPU.cuh"

#include <cstdint>

static constexpr auto MAX_SEEDSTRIPS = 200000;
static constexpr auto kClusterMaxStrips = 16;

typedef struct {
  stripgpu::detId_t *detId;
  stripgpu::stripId_t *stripId;
  uint8_t *adc;
  stripgpu::fedId_t *fedId;
  stripgpu::fedCh_t *fedCh;
  int *seedStripsNCIndex, *seedStripsMask, *seedStripsNCMask, *prefixSeedStripsNCMask;
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;
  int nSeedStripsNC;
  int nStrips;
} sst_data_t;

typedef struct {
  int *clusterLastIndexLeft, *clusterLastIndexRight;
  uint8_t *clusterADCs;
  bool *trueCluster;
  float *barycenter;
} clust_data_t;

void allocateSSTDataGPU(int max_strips, StripDataGPU& stripdata, sst_data_t *sst_data_d, sst_data_t **pt_sst_data_d, cudaStream_t stream);
void allocateClustDataGPU(int max_strips, clust_data_t *clust_data_d, clust_data_t **pt_clust_data_t, cudaStream_t stream);
void allocateClustData(int max_seedstrips, clust_data_t *clust_data, cudaStream_t stream);

void freeSSTDataGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, cudaStream_t stream);
void freeClustDataGPU(clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, cudaStream_t stream);
void freeClustData(clust_data_t *clust_data_t);

void cpyGPUToCPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, clust_data_t *clust_data, clust_data_t *clust_data_d, cudaStream_t stream);
void cpySSTDataToGPU(sst_data_t *sst_data, sst_data_t *sst_data_d, cudaStream_t stream);

void findClusterGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, const SiStripConditionsGPU *conditions, clust_data_t *clust_data_d, clust_data_t *pt_clust_data_d, cudaStream_t stream);

void setSeedStripsNCIndexGPU(sst_data_t *sst_data_d, sst_data_t *pt_sst_data_d, const SiStripConditionsGPU *conditions, cudaStream_t stream);

#endif