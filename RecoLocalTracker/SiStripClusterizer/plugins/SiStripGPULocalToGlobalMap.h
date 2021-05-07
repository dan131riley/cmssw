#ifndef RecoLocalTracker_SiStripClusterizer_SiStripGPULocalToGlobalMap_h
#define RecoLocalTracker_SiStripClusterizer_SiStripGPULocalToGlobalMap_h

#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"

class GeometricDet;
class MagneticField;
class SiStripBackPlaneCorrection;
class SiStripLorentzAngle;
class TrackerGeometry;
struct LocalToGlobalMap;

class SiStripGPULocalToGlobalMap {
public:
  using stripunit_t = std::tuple<unsigned int,
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
                                 float>;
  using stripunit_v = std::vector<stripunit_t>;

  SiStripGPULocalToGlobalMap(const GeometricDet& GeomDet2,
                             const MagneticField& magField,
                             const SiStripBackPlaneCorrection& backPlane,
                             const SiStripLorentzAngle& lorentz,
                             const TrackerGeometry& tkGx);
  ~SiStripGPULocalToGlobalMap();

  LocalToGlobalMap const* getGPUProductAsync(cudaStream_t stream) const;

  static constexpr int DETS_barrel = 7932;
  static constexpr int DETS_endcap = 7216;

private:
  void loadBarrel(const std::vector<const GeometricDet*>& dets_barrel,
                  const SiStripBackPlaneCorrection& BackPlaneCorrectionMap,
                  const MagneticField& MagFieldMap,
                  const SiStripLorentzAngle& LorentzAngleMap,
                  const stripunit_v& stripUnit);
  void loadEndcap(const std::vector<const GeometricDet*>& dets_endcap,
                  const SiStripBackPlaneCorrection& BackPlaneCorrectionMap,
                  const MagneticField& MagFieldMap,
                  const SiStripLorentzAngle& LorentzAngleMap,
                  const stripunit_v& stripUnit);

  // Holds the data in pinned CPU memory
  LocalToGlobalMap* localToGlobalMap_ = nullptr;

  // Helper struct to hold all information that has to be allocated and
  // deallocated per device
  struct GPUData {
    // Destructor should free all member pointers
    ~GPUData();
    LocalToGlobalMap* localToGlobalMapDevice_ = nullptr;
  };

  // Helper that takes care of complexity of transferring the data to
  // multiple devices
  cms::cuda::ESProduct<GPUData> gpuData_;
};

#endif
