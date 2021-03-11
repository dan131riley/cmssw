#include <iostream>
#include <cassert>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "SiStripRawToClusterGPUKernel.h"
#include "ChanLocsGPU.h"
//#include "unpackGPU.cuh"

namespace stripgpu {
  __global__ static void unpackChannels(const ChanLocStruct* chanlocs,
                                        const SiStripConditionsGPU* conditions,
                                        uint8_t* alldata,
                                        uint16_t* channel,
                                        stripgpu::stripId_t* stripId) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;

    const auto chan = nthreads * bid + tid;
    if (chan < chanlocs->size()) {
      const auto fedid = chanlocs->fedID(chan);
      const auto fedch = chanlocs->fedCh(chan);
      const auto ipair = conditions->iPair(fedid, fedch);
      const auto ipoff = kStripsPerChannel * ipair;

      const auto data = chanlocs->input(chan);
      const auto len = chanlocs->length(chan);

      if (data != nullptr && len > 0) {
        auto aoff = chanlocs->offset(chan);
        auto choff = chanlocs->inoff(chan);
        const auto end = choff + len;

        while (choff < end) {
          auto stripIndex = data[(choff++) ^ 7] + ipoff;
          const auto groupLength = data[(choff++) ^ 7];

          for (auto i = 0; i < 2; ++i) {
            stripId[aoff] = stripgpu::invStrip;
            alldata[aoff++] = 0;
          }

          for (auto i = 0; i < groupLength; ++i) {
            stripId[aoff] = stripIndex++;
            channel[aoff] = chan;
            alldata[aoff++] = data[(choff++) ^ 7];
          }
        }
      }
    }
  }

  StripDataGPU::StripDataGPU(size_t size, cudaStream_t stream) {
    alldataGPU_ = cms::cuda::make_device_unique<uint8_t[]>(size, stream);
    channelGPU_ = cms::cuda::make_device_unique<uint16_t[]>(size, stream);
    stripIdGPU_ = cms::cuda::make_device_unique<stripgpu::stripId_t[]>(size, stream);
  }

  void SiStripRawToClusterGPUKernel::unpackChannelsGPU(const SiStripConditionsGPU* conditions, cudaStream_t stream) {
    constexpr int nthreads = 128;
    const auto channels = chanlocsGPU_->size();
    const auto nblocks = (channels + nthreads - 1) / nthreads;

    unpackChannels<<<nblocks, nthreads, 0, stream>>>(chanlocsGPU_->chanLocStruct(),
                                                     conditions,
                                                     stripdata_->alldataGPU_.get(),
                                                     stripdata_->channelGPU_.get(),
                                                     stripdata_->stripIdGPU_.get());
  }
}  // namespace stripgpu
