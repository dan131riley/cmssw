#include <cassert>

#include <cuda_runtime.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "ChanLocsGPU.h"

ChannelLocs::ChannelLocs(size_t size, cudaStream_t stream)
  : ChannelLocsBase(size)
{
  if (size > 0) {
    input_ = cms::cuda::make_host_unique<const uint8_t*[]>(size, stream);
    inoff_ = cms::cuda::make_host_unique<size_t[]>(size, stream);
    offset_ = cms::cuda::make_host_unique<size_t[]>(size, stream);
    length_ = cms::cuda::make_host_unique<uint16_t[]>(size, stream);
    fedID_ = cms::cuda::make_host_unique<stripgpu::fedId_t[]>(size, stream);
    fedCh_ = cms::cuda::make_host_unique<stripgpu::fedCh_t[]>(size, stream);
  }
}

ChannelLocs::~ChannelLocs() {}

void ChanLocStruct::Fill(const ChannelLocsGPU& c)
{
  input_ = c.input();
  inoff_ = c.inoff();
  offset_ = c.offset();
  length_ = c.length();
  fedID_ = c.fedID();
  fedCh_ = c.fedCh();
  size_ = c.size();
}

ChannelLocsGPU::ChannelLocsGPU(size_t size, cudaStream_t stream)
  : ChannelLocsBase(size)
{
  if (size > 0) {
    input_ = cms::cuda::make_device_unique<const uint8_t*[]>(size, stream);
    inoff_ = cms::cuda::make_device_unique<size_t[]>(size, stream);
    offset_ = cms::cuda::make_device_unique<size_t[]>(size, stream);
    length_ = cms::cuda::make_device_unique<uint16_t[]>(size, stream);
    fedID_ = cms::cuda::make_device_unique<stripgpu::fedId_t[]>(size, stream);
    fedCh_ = cms::cuda::make_device_unique<stripgpu::fedCh_t[]>(size, stream);

    ChanLocStruct chanstruct;
    chanstruct.Fill(*this);
    chanstruct_ = cms::cuda::make_device_unique<ChanLocStruct>(stream);
    cudaCheck(cudaMemcpyAsync(chanstruct_.get(), &chanstruct, sizeof(ChanLocStruct), cudaMemcpyDefault, stream));
  }
}

void ChannelLocsGPU::reset(const ChannelLocs& c, const std::vector<uint8_t*>& inputGPU, cudaStream_t stream)
{
  assert(c.size() == size_);
  cudaCheck(cudaMemcpyAsync(input_.get(), inputGPU.data(), sizeof(uint8_t*)*size_, cudaMemcpyDefault, stream));
  cms::cuda::copyAsync(inoff_, c.inoff_, size_, stream);
  cms::cuda::copyAsync(offset_, c.offset_, size_, stream);
  cms::cuda::copyAsync(length_, c.length_, size_, stream);
  cms::cuda::copyAsync(fedID_, c.fedID_, size_, stream);
  cms::cuda::copyAsync(fedCh_, c.fedCh_, size_, stream);
}

ChannelLocsGPU::~ChannelLocsGPU()
{
}
