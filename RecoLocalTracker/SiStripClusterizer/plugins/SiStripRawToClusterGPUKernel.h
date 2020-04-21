#ifndef RecoLocalTracker_SiStripClusterizer_plugins_SiStripRawToClusterGPUKernel_h
#define RecoLocalTracker_SiStripClusterizer_plugins_SiStripRawToClusterGPUKernel_h

//#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "ChanLocsGPU.h"
#include "clusterGPU.cuh"

#include <cuda_runtime.h>

#include <vector>
#include <memory>

class SiStripConditionsGPUWrapper;
class FEDRawData;
namespace sistrip {
  class FEDBuffer;
}

class StripDataGPU;
class sst_data_t;
class clust_data_t;

namespace stripgpu {
  class SiStripRawToClusterGPUKernel {
  public:
    void makeAsync(const std::vector<const FEDRawData*>& rawdata,
                   const std::vector<std::unique_ptr<sistrip::FEDBuffer>>& buffers,
                   const SiStripConditionsGPUWrapper* conditionswrapper,
                   cudaStream_t stream);
    std::unique_ptr<edmNew::DetSetVector<SiStripCluster>> getResults(cudaStream_t stream);

  private:
    void reset();

    cms::cuda::device::unique_ptr<uint8_t[]> fedRawDataGPU;
    std::unique_ptr<StripDataGPU> stripdata;

    std::unique_ptr<ChannelLocs> chanlocs;
    std::unique_ptr<ChannelLocsGPU> chanlocsGPU;

    cms::cuda::host::unique_ptr<sst_data_t> sst_data_d;
    sst_data_t *pt_sst_data_d;

    std::unique_ptr<clust_data_t> clust_data_d;
    std::unique_ptr<clust_data_t> clust_data;
    clust_data_t *pt_clust_data_d;
  };
}
#endif
