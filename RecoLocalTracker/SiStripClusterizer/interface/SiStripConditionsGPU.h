#ifndef RecoLocalTracker_SiStripClusterizer_SiStripConditionsGPU_h
#define RecoLocalTracker_SiStripClusterizer_SiStripConditionsGPU_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "CUDADataFormats/SiStripCluster/interface/GPUtypes.h"

namespace stripgpu {
  static constexpr int kStripsPerChannel = 256;
  static constexpr int kFedFirst = 50;
  static constexpr int kFedLast = 489;
  static constexpr int kFedCount = kFedLast - kFedFirst + 1;
  static constexpr int kChannelCount = 96;
  static constexpr int kStripsPerFed = kChannelCount*kStripsPerChannel;

  __host__ __device__ inline fedId_t fedIndex(fedId_t fed) { return fed-kFedFirst; }
  __host__ __device__ inline stripId_t stripIndex(fedCh_t channel, stripId_t strip) {
    return channel*kStripsPerChannel + (strip % kStripsPerChannel);
  }
}

struct SiStripConditionsGPU {
  __host__ __device__ void setStrip(stripgpu::fedId_t fed,
                                    stripgpu::fedCh_t channel,
                                    stripgpu::stripId_t strip,
                                    float noise,
                                    float gain,
                                    bool bad)
  {
    noise_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)] = noise;
    gain_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)] = gain;
    bad_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)] = bad;
  }

  __host__ __device__ void setInvThickness(stripgpu::fedId_t fed, stripgpu::fedCh_t channel, float invthick)
  {
    invthick_[stripgpu::fedIndex(fed)][channel] = invthick;
  }

  __host__ __device__ stripgpu::detId_t detID(stripgpu::fedId_t fed, stripgpu::fedCh_t channel) const
  { return detID_[stripgpu::fedIndex(fed)][channel]; }

  __host__ __device__ stripgpu::APVPair_t iPair(stripgpu::fedId_t fed, stripgpu::fedCh_t channel) const
  { return iPair_[stripgpu::fedIndex(fed)][channel]; }

  __host__ __device__ float invthick(stripgpu::fedId_t fed, stripgpu::fedCh_t channel) const
  { return invthick_[stripgpu::fedIndex(fed)][channel]; }

  __host__ __device__ float noise(stripgpu::fedId_t fed, stripgpu::fedCh_t channel, stripgpu::stripId_t strip) const
  { return noise_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)]; }

  __host__ __device__ float gain(stripgpu::fedId_t fed, stripgpu::fedCh_t channel, stripgpu::stripId_t strip) const
  { return gain_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)]; }

  __host__ __device__ bool bad(stripgpu::fedId_t fed, stripgpu::fedCh_t channel, stripgpu::stripId_t strip) const
  { return bad_[stripgpu::fedIndex(fed)][stripgpu::stripIndex(channel, strip)]; }

  alignas(128) float noise_[stripgpu::kFedCount][stripgpu::kStripsPerFed];
  alignas(128) float gain_[stripgpu::kFedCount][stripgpu::kStripsPerFed];
  alignas(128) bool bad_[stripgpu::kFedCount][stripgpu::kStripsPerFed];
  alignas(128) float invthick_[stripgpu::kFedCount][stripgpu::kChannelCount];
  alignas(128) stripgpu::detId_t detID_[stripgpu::kFedCount][stripgpu::kChannelCount];
  alignas(128) stripgpu::APVPair_t iPair_[stripgpu::kFedCount][stripgpu::kChannelCount];
};

#endif
