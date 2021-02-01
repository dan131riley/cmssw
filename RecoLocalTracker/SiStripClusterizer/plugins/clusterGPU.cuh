#ifndef _CLUSTER_GPU_KERNEL_
#define _CLUSTER_GPU_KERNEL_

#include "CUDADataFormats/SiStripCluster/interface/GPUtypes.h"

#include <cstdint>

namespace stripgpu {
  class StripDataGPU;
}
struct ChanLocStruct;

static constexpr auto MAX_SEEDSTRIPS = 200000;
static constexpr uint32_t kClusterMaxStrips = SiStripClustersCUDA::kClusterMaxStrips;

struct sst_data_t {
  const ChanLocStruct* chanlocs;
  uint8_t *adc;
  uint16_t* channel;
  stripgpu::stripId_t *stripId;
  int *seedStripsNCIndex, *seedStripsMask, *seedStripsNCMask, *prefixSeedStripsNCMask;
  int nSeedStripsNC;
  int nStrips;
};
#endif
