#ifndef RecoLocalTracker_SiStripClusterizer_plugins_MkFitSiStripHitGPUKernel_h
#define RecoLocalTracker_SiStripClusterizer_plugins_MkFitSiStripHitGPUKernel_h

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "clusterGPU.cuh"
#include "CUDADataFormats/SiStripCluster/interface/MkFitSiStripClustersCUDA.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include <cuda_runtime.h>

#include <vector>
#include <memory>


namespace stripgpu {

  class MkFitSiStripHitGPUKernel {
  public:
    void loadBarrel(const std::vector<const GeometricDet*> dets_barrel, /*const std::vector<const GeomDet*> rots_barrel,*/const SiStripBackPlaneCorrection* BackPlaneCorrectionMap,const MagneticField* MagFieldMap,const SiStripLorentzAngle* LorentzAngleMap,const std::vector<std::tuple<unsigned int,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>> stripUnit);
    void loadEndcap(const std::vector<const GeometricDet*> dets_endcap, /*const std::vector<const GeomDet*> rots_endcap,*/const SiStripBackPlaneCorrection* BackPlaneCorrectionMap,const MagneticField* MagFieldMap,const SiStripLorentzAngle* LorentzAngleMap,const std::vector<std::tuple<unsigned int,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>> stripUnit);
    void makeGlobal(SiStripClustersCUDA& clusters_d,MkFitSiStripClustersCUDA& clusters_g, cudaStream_t stream);
  };

}
#endif
