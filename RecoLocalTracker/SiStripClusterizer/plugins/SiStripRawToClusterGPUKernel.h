#ifndef RecoLocalTracker_SiStripClusterizer_plugins_SiStripRawToClusterGPUKernel_h
#define RecoLocalTracker_SiStripClusterizer_plugins_SiStripRawToClusterGPUKernel_h

//#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include <cuda_runtime.h>

#include <vector>
#include <memory>

class SiStripConditionsGPUWrapper;
class FEDRawData;
namespace sistrip {
  class FEDBuffer;
}

namespace stripgpu {
  class SiStripRawToClusterGPUKernel {
  public:
    void makeAsync(const std::vector<const FEDRawData*>& rawdata,
                   const std::vector<std::unique_ptr<sistrip::FEDBuffer>>& buffers,
                   const SiStripConditionsGPUWrapper* conditionswrapper,
                   cudaStream_t stream);
    void getResults();
  private:
  };
}

#endif
