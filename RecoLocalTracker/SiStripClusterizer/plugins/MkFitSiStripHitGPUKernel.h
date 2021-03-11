#ifndef RecoLocalTracker_SiStripClusterizer_plugins_MkFitSiStripHitGPUKernel_h
#define RecoLocalTracker_SiStripClusterizer_plugins_MkFitSiStripHitGPUKernel_h

#include <cuda_runtime.h>
#include <vector>
#include <memory>

class GeometricDet;
class SiStripBackPlaneCorrection;
class MagneticField;
class SiStripLorentzAngle;
class SiStripClustersCUDA;
class MkFitSiStripClustersCUDA;
struct LocalToGlobalMap;

namespace stripgpu {

  class MkFitSiStripHitGPUKernel {
  public:
    MkFitSiStripHitGPUKernel();
    ~MkFitSiStripHitGPUKernel();
    void loadBarrel(
        const std::vector<const GeometricDet*> dets_barrel,
        /*const std::vector<const GeomDet*> rots_barrel,*/ const SiStripBackPlaneCorrection* BackPlaneCorrectionMap,
        const MagneticField* MagFieldMap,
        const SiStripLorentzAngle* LorentzAngleMap,
        const std::vector<std::tuple<unsigned int,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float>> stripUnit);
    void loadEndcap(
        const std::vector<const GeometricDet*> dets_endcap,
        /*const std::vector<const GeomDet*> rots_endcap,*/ const SiStripBackPlaneCorrection* BackPlaneCorrectionMap,
        const MagneticField* MagFieldMap,
        const SiStripLorentzAngle* LorentzAngleMap,
        const std::vector<std::tuple<unsigned int,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float,
                                     float>> stripUnit);
    void makeGlobal(SiStripClustersCUDA& clusters_d, MkFitSiStripClustersCUDA& clusters_g, cudaStream_t stream);
    const LocalToGlobalMap* toDevice();

  private:
    LocalToGlobalMap* localToGlobalMap_h = nullptr;
    LocalToGlobalMap* localToGlobalMap_d = nullptr;
  };

}  // namespace stripgpu
#endif
