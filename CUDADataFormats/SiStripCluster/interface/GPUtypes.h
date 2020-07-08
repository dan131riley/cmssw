#ifndef CUDADataFormats_SiStripCluster_interface_GPUtypes_h
#define CUDADataFormats_SiStripCluster_interface_GPUtypes_h

#include <cstdint>
#include <limits>

namespace stripgpu {
  using detId_t = std::uint32_t;
  using fedId_t = std::uint16_t;
  using fedCh_t = std::uint8_t;
  using APVPair_t = std::uint16_t;
  using stripId_t = std::uint16_t;

  static constexpr detId_t invDet = std::numeric_limits<detId_t>::max();
  static constexpr fedId_t invFed = std::numeric_limits<fedId_t>::max();
  static constexpr stripId_t invStrip = std::numeric_limits<stripId_t>::max();
}

#endif
