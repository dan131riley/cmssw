#ifndef RecoLocalTracker_SiStripClusterizer_plugins_SiStripRawToClusterGPUKernel_h
#define RecoLocalTracker_SiStripClusterizer_plugins_SiStripRawToClusterGPUKernel_h

//#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
//#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
//#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include "SiStripConditionsGPU.h"
//#include "ChanLocsGPU.h"
#include "clusterGPU.cuh"

#include <cuda_runtime.h>

#include <vector>
#include <memory>

class SiStripConditionsGPUWrapper;
class ChannelLocs;
class ChannelLocsGPU;
class FEDRawData;
namespace sistrip {
  class FEDBuffer;
}

class sst_data_t;
class clust_data_t;

namespace stripgpu {
  class StripDataGPU {
  public:
    StripDataGPU(size_t size, cudaStream_t stream);

    cms::cuda::device::unique_ptr<uint8_t[]> alldataGPU_;
    cms::cuda::device::unique_ptr<uint16_t[]> channelGPU_;
    cms::cuda::device::unique_ptr<stripgpu::stripId_t[]> stripIdGPU_;
    cms::cuda::device::unique_ptr<int[]> seedStripsMask_;
    cms::cuda::device::unique_ptr<int[]> prefixSeedStripsNCMask_;
  };

  class SiStripRawToClusterGPUKernel {
  public:
    SiStripRawToClusterGPUKernel()
      : fedIndex_(stripgpu::kFedCount, stripgpu::invFed)
    {
      fedRawDataOffsets_.reserve(stripgpu::kFedCount);
    }
    void makeAsync(const std::vector<const FEDRawData*>& rawdata,
                   const std::vector<std::unique_ptr<sistrip::FEDBuffer>>& buffers,
                   const SiStripConditionsGPUWrapper& conditionswrapper,
                   cudaStream_t stream);
    void copyAsync(cudaStream_t stream);
    SiStripClustersCUDA getResults(cudaStream_t stream);

  private:
    void reset();
    void unpackChannelsGPU(const SiStripConditionsGPU* conditions, cudaStream_t stream);
    void allocateSSTDataGPU(int max_strips, cudaStream_t stream);
    void freeSSTDataGPU(cudaStream_t stream);

    void setSeedStripsNCIndexGPU(const SiStripConditionsGPU *conditions, cudaStream_t stream);
    void findClusterGPU(const SiStripConditionsGPU *conditions, cudaStream_t stream);

    std::vector<stripgpu::fedId_t> fedIndex_;
    std::vector<size_t> fedRawDataOffsets_;

    cms::cuda::host::unique_ptr<uint8_t[]> fedRawDataHost_;
    std::unique_ptr<StripDataGPU> stripdata_;

    std::unique_ptr<ChannelLocs> chanlocs_;
    std::unique_ptr<ChannelLocsGPU> chanlocsGPU_;

    cms::cuda::host::unique_ptr<sst_data_t> sst_data_d_;
    cms::cuda::device::unique_ptr<sst_data_t> pt_sst_data_d_;

    SiStripClustersCUDA clusters_d_;
    //std::unique_ptr<clust_data_t> clust_data_d;
    //std::unique_ptr<clust_data_t> clust_data;
    //clust_data_t *pt_clust_data_d;
  };
}
#endif
