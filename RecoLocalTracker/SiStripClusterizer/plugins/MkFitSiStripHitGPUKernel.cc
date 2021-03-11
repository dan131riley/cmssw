#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"

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

#include "MkFitSiStripHitGPUKernel.h"
#include "localToGlobal.cuh"

namespace stripgpu {
  MkFitSiStripHitGPUKernel::MkFitSiStripHitGPUKernel() {
    cudaCheck(cudaMallocHost(&localToGlobalMap_h, sizeof(localToGlobalMap)));
  }

  MkFitSiStripHitGPUKernel::~MkFitSiStripHitGPUKernel() {
    if (localToGlobalMap_h) {
      cudaCheck(cudaFreeHost(localToGlobalMap_h));
    }
    if (localToGlobalMap_d) {
      cudaCheck(cudaFree(localToGlobalMap_d));
    }
  }

  void MkFitSiStripHitGPUKernel::loadBarrel(
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
                                   float>> stripUnit) {
    printf("LOADING BARREL\n");
    auto indexer3_h = localToGlobalMap_h->indexer3;
    auto indexer5_h = localToGlobalMap_h->indexer5;

    cudaMemset(indexer3_h, -1, sizeof(LocalToGlobalMap_h::indexer3));
    cudaMemset(indexer5_h, -1, sizeof(LocalToGlobalMap_h::indexer5));

    int* det_num_h = localToGlobalMap_h->det_num_bd;
    double* pitch_h = localToGlobalMap_h->pitch_bd;
    double* offset_h = localToGlobalMap_h->offset_bd;
    double* len_h = localToGlobalMap_h->len_bd;
    double* pos_x_h = localToGlobalMap_h->pos_x_bd;
    double* pos_y_h = localToGlobalMap_h->pos_y_bd;
    double* pos_z_h = localToGlobalMap_h->pos_z_bd;
    double* R11_h = localToGlobalMap_h->R11_bd;
    double* R12_h = localToGlobalMap_h->R12_bd;
    double* R13_h = localToGlobalMap_h->R13_bd;
    double* R21_h = localToGlobalMap_h->R21_bd;
    double* R22_h = localToGlobalMap_h->R22_bd;
    double* R23_h = localToGlobalMap_h->R23_bd;
    double* backPlane_h = localToGlobalMap_h->backPlane_bd;
    double* thickness_h = localToGlobalMap_h->thickness_bd;
    double* drift_x_h = localToGlobalMap_h->drift_x_bd;

    for (auto it = dets_barrel.begin(); it != dets_barrel.end(); ++it) {
      int i = std::distance(dets_barrel.begin(), it);
      const GeometricDet* det = dets_barrel[i];
      det_num_h[i] = det->geographicalId().rawId();

      int nstrip = int(128 * det->siliconAPVNum());
      std::unique_ptr<const Bounds> bounds(det->bounds());
      len_h[i] = bounds->length();
      double width = bounds->width();
      thickness_h[i] = bounds->thickness();
      pitch_h[i] = width / nstrip;
      offset_h[i] = -0.5 * width;
      int sub = (det_num_h[i] >> 25) & 0x7;
      if (sub == 3) {
        indexer3_h[index_lookup3(det_num_h[i])] = i;
      } else if (sub == 5) {
        indexer5_h[index_lookup5(det_num_h[i])] = i;
      }
    }

    for (auto it = stripUnit.begin(); it != stripUnit.end(); ++it) {
      int j = std::distance(stripUnit.begin(), it);
      const auto dus = stripUnit[j];

      auto rot_num = std::get<0>(dus);
      int i = -1;
      int sub = (rot_num >> 25) & 0x7;
      if (sub == 3) {
        int lookup = index_lookup3(rot_num);
        if (lookup > 31214) {
          continue;
        }
        i = indexer3_h[lookup];
      }
      if (sub == 5) {
        int lookup = index_lookup5(rot_num);
        if (lookup > 46426) {
          continue;
        }
        i = indexer5_h[lookup];
      }
      if (i == -1) {
        continue;
      }
      backPlane_h[i] = BackPlaneCorrectionMap->getBackPlaneCorrection(rot_num);
      double lorentzAngle = LorentzAngleMap->getLorentzAngle(rot_num);
      drift_x_h[i] = -lorentzAngle * std::get<2>(dus);
      pos_x_h[i] = std::get<4>(dus);
      pos_y_h[i] = std::get<5>(dus);
      pos_z_h[i] = std::get<6>(dus);
      R11_h[i] = std::get<7>(dus);
      R12_h[i] = std::get<8>(dus);
      R13_h[i] = std::get<9>(dus);
      R21_h[i] = std::get<10>(dus);
      R22_h[i] = std::get<11>(dus);
      R23_h[i] = std::get<12>(dus);
    }
  }

  void MkFitSiStripHitGPUKernel::loadEndcap(
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
                                   float>> stripUnit) {
    printf("LOADING EndCap\n");
    short* indexer4_h = localToGlobalMap_h->indexer4;
    short* indexer6_h = localToGlobalMap_h->indexer6;
    cudaMemset(indexer4_h, -1, sizeof(LocalToGlobalMap::indexer4));
    cudaMemset(indexer6_h, -1, sizeof(LocalToGlobalMap::indexer6));

    int* det_num_h = localToGlobalMap_h->det_num_ed;
    int* yAx_h = localToGlobalMap_h->yAx_ed;
    double* backPlane_h = localToGlobalMap_h->backPlane_ed;
    double* thickness_h = localToGlobalMap_h->thickness_ed;
    double* drift_x_h = localToGlobalMap_h->drift_x_ed;
    double* drift_y_h = localToGlobalMap_h->drift_y_ed;
    double* rCross_h = localToGlobalMap_h->rCross_ed;
    double* aw_h = localToGlobalMap_h->aw_ed;
    double* phi_h = localToGlobalMap_h->phi_ed;
    double* len_h = localToGlobalMap_h->len_ed;
    double* pos_x_h = localToGlobalMap_h->pos_x_ed;
    double* pos_y_h = localToGlobalMap_h->pos_y_ed;
    double* pos_z_h = localToGlobalMap_h->pos_z_ed;
    double* R11_h = localToGlobalMap_h->R11_ed;
    double* R12_h = localToGlobalMap_h->R12_ed;
    double* R13_h = localToGlobalMap_h->R13_ed;
    double* R21_h = localToGlobalMap_h->R21_ed;
    double* R22_h = localToGlobalMap_h->R22_ed;
    double* R23_h = localToGlobalMap_h->R23_ed;

    for (auto it = dets_endcap.begin(); it != dets_endcap.end(); ++it) {
      int i = std::distance(dets_endcap.begin(), it);
      const GeometricDet* det = dets_endcap[i];
      det_num_h[i] = det->geographicalId().rawId();

      int nstrip = int(128 * det->siliconAPVNum());
      std::unique_ptr<const Bounds> bounds(det->bounds());
      yAx_h[i] = (dynamic_cast<const TrapezoidalPlaneBounds*>(&(*bounds)))->yAxisOrientation();
      len_h[i] = bounds->length();
      thickness_h[i] = bounds->thickness();
      float width = bounds->width();
      float w_halfl = bounds->widthAtHalfLength();
      rCross_h[i] = w_halfl * len_h[i] / (2 * (width - w_halfl));
      aw_h[i] = atan2(w_halfl / 2., static_cast<float>(rCross_h[i])) / (nstrip / 2);
      phi_h[i] = -(0.5 * nstrip) * aw_h[i];

      int sub = (det_num_h[i] >> 25) & 0x7;
      if (sub == 4) {
        indexer4_h[index_lookup4(det_num_h[i])] = i;
      } else if (sub == 6) {
        indexer6_h[index_lookup6(det_num_h[i])] = i;
      }
    }
    for (auto it = stripUnit.begin(); it != stripUnit.end(); ++it) {
      int j = std::distance(stripUnit.begin(), it);
      const auto dus = stripUnit[j];

      auto rot_num = std::get<0>(dus);
      int i = -1;
      int sub = (rot_num >> 25) & 0x7;
      if (sub == 4) {
        int lookup = index_lookup4(rot_num);
        if (lookup > 16208) {
          continue;
        }
        i = indexer4_h[lookup];
      }
      if (sub == 6) {
        int lookup = index_lookup6(rot_num);
        if (lookup > 145652) {
          continue;
        }
        i = indexer6_h[lookup];
      }
      if (i == -1) {
        continue;
      }
      backPlane_h[i] = BackPlaneCorrectionMap->getBackPlaneCorrection(rot_num);
      double lorentzAngle = LorentzAngleMap->getLorentzAngle(rot_num);
      drift_x_h[i] = -lorentzAngle * std::get<2>(dus);
      drift_y_h[i] = lorentzAngle * std::get<1>(dus);
      pos_x_h[i] = std::get<4>(dus);
      pos_y_h[i] = std::get<5>(dus);
      pos_z_h[i] = std::get<6>(dus);
      R11_h[i] = std::get<7>(dus);
      R12_h[i] = std::get<8>(dus);
      R13_h[i] = std::get<9>(dus);
      R21_h[i] = std::get<10>(dus);
      R22_h[i] = std::get<11>(dus);
      R23_h[i] = std::get<12>(dus);
    }
  }
  const LocalToGlobalMap* MkFitSiStripHitGPUKernel::toDevice() {
    cudaCheck(cudaMalloc(&localToGlobalMap_d, sizeof(localToGlobalMap)));
    cudaCheck(cudaMemcpy(localToGlobalMap_d, localToGlobalMap_h, cudaMemcpyDefault));
    return localToGlobalMap_d;
  }
}  // namespace stripgpu
