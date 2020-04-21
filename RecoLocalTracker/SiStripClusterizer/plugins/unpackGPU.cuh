#pragma once
//#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripConditionsGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

class SiStripConditionsGPU;
class ChannelLocsGPU;

class StripDataGPU {
public:
  StripDataGPU(size_t size, cudaStream_t stream);

  cms::cuda::device::unique_ptr<uint8_t[]> alldataGPU_;
  cms::cuda::device::unique_ptr<stripgpu::detId_t[]> detIdGPU_;
  cms::cuda::device::unique_ptr<stripgpu::stripId_t[]> stripIdGPU_;
  cms::cuda::device::unique_ptr<stripgpu::fedId_t[]> fedIdGPU_;
  cms::cuda::device::unique_ptr<stripgpu::fedCh_t[]> fedChGPU_;
};

void unpackChannelsGPU(const ChannelLocsGPU* chanlocs, const SiStripConditionsGPU* conditions, StripDataGPU* stripdata, cudaStream_t stream);
