#ifndef RecoLocalTracker_SiStripClusterizer_SiStripConditionsGPUWrapper_h
#define RecoLocalTracker_SiStripClusterizer_SiStripConditionsGPUWrapper_h

#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripConditionsGPU.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"

class DetToFed {
public:
  DetToFed(stripgpu::detId_t detid, stripgpu::APVPair_t ipair, stripgpu::fedId_t fedid, stripgpu::fedCh_t fedch)
    : detid_(detid), ipair_(ipair), fedid_(fedid), fedch_(fedch) {}
  stripgpu::detId_t detID() const { return detid_; }
  stripgpu::APVPair_t pair() const { return ipair_; }
  stripgpu::fedId_t fedID() const { return fedid_; }
  stripgpu::fedCh_t fedCh() const { return fedch_; }
private:
  stripgpu::detId_t detid_;
  stripgpu::APVPair_t ipair_;
  stripgpu::fedId_t fedid_;
  stripgpu::fedCh_t fedch_;
};
using DetToFeds = std::vector<DetToFed>;

class SiStripConditionsGPU;

class SiStripConditionsGPUWrapper {
public:
  SiStripConditionsGPUWrapper(const StripClusterizerAlgorithm* clusterizer);
  ~SiStripConditionsGPUWrapper();

  // Function to return the actual payload on the memory of the current device
  SiStripConditionsGPU const *getGPUProductAsync(cudaStream_t stream) const;

  const DetToFeds& detToFeds() const { return detToFeds_; }
private:
  // Holds the data in pinned CPU memory
  SiStripConditionsGPU *conditions_ = nullptr;

  // Helper struct to hold all information that has to be allocated and
  // deallocated per device
  struct GPUData {
    // Destructor should free all member pointers
    ~GPUData();
    SiStripConditionsGPU *conditionsDevice = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
  DetToFeds detToFeds_;
};

#endif
