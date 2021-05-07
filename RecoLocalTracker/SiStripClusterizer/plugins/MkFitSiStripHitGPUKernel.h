#ifndef RecoLocalTracker_SiStripClusterizer_plugins_MkFitSiStripHitGPUKernel_h
#define RecoLocalTracker_SiStripClusterizer_plugins_MkFitSiStripHitGPUKernel_h

#include <cuda_runtime.h>
#include <vector>
#include <memory>

class SiStripClustersCUDA;
class MkFitSiStripClustersCUDA;
struct LocalToGlobalMap;

namespace stripgpu {

  class MkFitSiStripHitGPUKernel {
  public:
    void makeGlobal(const SiStripClustersCUDA& clusters_d,
                    const LocalToGlobalMap* geometry,
                    MkFitSiStripClustersCUDA& clusters_g,
                    cudaStream_t stream);
  };

}  // namespace stripgpu
#endif
