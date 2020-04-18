#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripConditionsGPUWrapper.h"

SiStripConditionsGPUWrapper::SiStripConditionsGPUWrapper(const StripClusterizerAlgorithm* clusterizer)
{
  cudaCheck(cudaMallocHost(&conditions_, sizeof(SiStripConditionsGPU)));
  detToFeds_.clear();

  for ( auto detID : clusterizer->allDetIds()) {
    auto det = clusterizer->findDetId(detID);
    for (auto const conn : clusterizer->currentConnection(det)) {
      if (conn && conn->fedId() && conn->isConnected()) {
        auto fedID = conn->fedId();
        auto fedCh = conn->fedCh();
        auto iPair = conn->apvPairNumber();

        detToFeds_.emplace_back(detID, iPair, fedID, fedCh);

        conditions_->detID_[stripgpu::fedIndex(fedID)][fedCh] = detID;
        conditions_->iPair_[stripgpu::fedIndex(fedID)][fedCh] = iPair;
        conditions_->setInvThickness(fedID, fedCh, siStripClusterTools::sensorThicknessInverse(detID));

        auto offset = 256 * iPair;

        for (auto strip = 0; strip < 256; ++strip) {
          auto detstrip = strip + offset;
          conditions_->setStrip(fedID, fedCh, strip, det.noise(detstrip), det.gain(detstrip), det.bad(detstrip));
        }
      }
    }
  }
  std::sort(detToFeds_.begin(), detToFeds_.end(),
    [](const DetToFed& a, const DetToFed& b){ 
      return a.detID() < b.detID() || (a.detID() == b.detID() && a.pair() < b.pair());
  });
}

SiStripConditionsGPUWrapper::~SiStripConditionsGPUWrapper() {
  if (nullptr != conditions_) {
    cudaCheck(cudaFreeHost(conditions_));
  }
}

SiStripConditionsGPU const *SiStripConditionsGPUWrapper::getGPUProductAsync(cudaStream_t stream) const {
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, cudaStream_t stream) {
    // Allocate the payload object on the device memory.
    cudaCheck(cudaMalloc(&data.conditionsDevice, sizeof(SiStripConditionsGPU)));
    cudaCheck(cudaMemcpyAsync(data.conditionsDevice, conditions_, sizeof(SiStripConditionsGPU), cudaMemcpyDefault, stream));
  });
  // Returns the payload object on the memory of the current device
  return data.conditionsDevice;
}

SiStripConditionsGPUWrapper::GPUData::~GPUData() {
  cudaCheck(cudaFree(conditionsDevice));
}
