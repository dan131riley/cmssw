#include <iostream>
#include <cassert>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "ChanLocsGPU.h"
#include "unpackGPU.cuh"

constexpr auto kStripsPerChannel = stripgpu::kStripsPerChannel;

__global__
static void unpackChannels(const ChanLocStruct* chanlocs, const SiStripConditionsGPU* conditions,
                           uint8_t* alldata, stripgpu::detId_t* detId, stripgpu::stripId_t* stripId,
                           stripgpu::fedId_t* fedId, stripgpu::fedCh_t* fedCh)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;

  const auto chan = nthreads*bid + tid;
  if (chan < chanlocs->size()) {
    const auto fedid = chanlocs->fedID(chan);
    const auto fedch = chanlocs->fedCh(chan);
    const auto detid = conditions->detID(fedid, fedch);
    const auto ipoff = kStripsPerChannel*conditions->iPair(fedid, fedch);

    const auto data = chanlocs->input(chan);
    const auto len = chanlocs->length(chan);

    if (data != nullptr && len > 0) {
      auto aoff = chanlocs->offset(chan);
      auto choff = chanlocs->inoff(chan);
      const auto end = aoff + len;

      while (aoff < end) {
        stripId[aoff] = stripgpu::invStrip;
        detId[aoff] = stripgpu::invDet;
        alldata[aoff] = data[(choff++)^7];
        auto stripIndex = alldata[aoff++] + ipoff;
 
        stripId[aoff] = stripgpu::invStrip;
        detId[aoff] = detid;
        alldata[aoff] = data[(choff++)^7];
        const auto groupLength = alldata[aoff++];

        for (auto i = 0; i < groupLength; ++i) {
          detId[aoff] = detid;
          fedId[aoff] = fedid;
          fedCh[aoff] = fedch;
          stripId[aoff] = stripIndex++;
          alldata[aoff++] = data[(choff++)^7];
        }
      }
    }
  }
}

StripDataGPU::StripDataGPU(size_t size, cudaStream_t stream)
{
  alldataGPU_ = cms::cuda::make_device_unique<uint8_t[]>(size, stream);
  detIdGPU_ = cms::cuda::make_device_unique<stripgpu::detId_t[]>(size, stream);
  stripIdGPU_ = cms::cuda::make_device_unique<stripgpu::stripId_t[]>(size, stream);
  fedIdGPU_ = cms::cuda::make_device_unique<stripgpu::fedId_t[]>(size, stream);
  fedChGPU_ = cms::cuda::make_device_unique<stripgpu::fedCh_t[]>(size, stream);
}

void unpackChannelsGPU(const ChannelLocsGPU* chanlocs, const SiStripConditionsGPU* conditions, StripDataGPU* stripdata, cudaStream_t stream)
{
  constexpr int nthreads = 128;
  const auto channels = chanlocs->size();
  const auto nblocks = (channels + nthreads - 1)/nthreads;
  
  unpackChannels<<<nblocks, nthreads, 0, stream>>>(chanlocs->chanLocStruct(), conditions,
                                                   stripdata->alldataGPU_.get(),
                                                   stripdata->detIdGPU_.get(),
                                                   stripdata->stripIdGPU_.get(),
                                                   stripdata->fedIdGPU_.get(),
                                                   stripdata->fedChGPU_.get());
}
