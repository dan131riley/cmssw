#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "SiStripGPULocalToGlobalMap.h"
#include "localToGlobal.cuh"

using stripgpu::index_lookup3;
using stripgpu::index_lookup4;
using stripgpu::index_lookup5;
using stripgpu::index_lookup6;

SiStripGPULocalToGlobalMap::SiStripGPULocalToGlobalMap(const GeometricDet& GeomDet2,
                                                       const MagneticField& MagFieldMap,
                                                       const SiStripBackPlaneCorrection& BackPlaneCorrectionMap,
                                                       const SiStripLorentzAngle& LorentzAngleMap,
                                                       const TrackerGeometry& tkG) {
  stripunit_v stripUnit;
  stripUnit.reserve(tkG.detUnits().size());

  for (auto&& dus : tkG.detUnits()) {
    auto&& rot_num = dus->geographicalId().rawId();
    auto&& magField = (dus->surface()).toLocal(MagFieldMap.inTesla(dus->surface().position()));

    Surface::RotationType rot = dus->surface().rotation();
    Surface::PositionType pos = dus->surface().position();

    stripUnit.emplace_back(rot_num,
                           magField.x(),
                           magField.y(),
                           magField.z(),
                           pos.x(),
                           pos.y(),
                           pos.z(),
                           rot.xx(),
                           rot.xy(),
                           rot.xz(),
                           rot.yx(),
                           rot.yy(),
                           rot.yz(),
                           rot.zx(),
                           rot.zy(),
                           rot.zz());
  }

  //sort the tracker geometry into barrel and endcap vectors
  std::vector<const GeometricDet*> dets_barrel;
  std::vector<const GeometricDet*> dets_endcap;

  dets_barrel.reserve(DETS_barrel);
  dets_endcap.reserve(DETS_endcap);

  for (auto& it : GeomDet2.deepComponents()) {
    DetId det = it->geographicalId();

    int subdet = det.subdetId();
    if (subdet == 3 || subdet == 5) {
      dets_barrel.emplace_back(it);
    } else if (subdet == 4 || subdet == 6) {
      dets_endcap.emplace_back(it);
    }
  }

  //sort and erase duplicates.
  auto detcomp = [](const GeometricDet* lhs, const GeometricDet* rhs) {
    DetId detl = lhs->geographicalId();
    DetId detr = rhs->geographicalId();
    return detl.rawId() < detr.rawId();
  };

  sort(dets_barrel.begin(), dets_barrel.end(), detcomp);
  sort(dets_endcap.begin(), dets_endcap.end(), detcomp);

  auto deteq = [](const GeometricDet* lhs, const GeometricDet* rhs) {
    DetId detl = lhs->geographicalId();
    DetId detr = rhs->geographicalId();
    return detl.rawId() == detr.rawId();
  };

  dets_barrel.erase(unique(dets_barrel.begin(), dets_barrel.end(), deteq), dets_barrel.end());
  dets_endcap.erase(unique(dets_endcap.begin(), dets_endcap.end(), deteq), dets_endcap.end());

  cudaCheck(cudaMallocHost(&localToGlobalMap_, sizeof(LocalToGlobalMap)));

  //Load the barrel and endcap geometry into pinned host memory
  loadBarrel(dets_barrel, BackPlaneCorrectionMap, MagFieldMap, LorentzAngleMap, stripUnit);
  loadEndcap(dets_endcap, BackPlaneCorrectionMap, MagFieldMap, LorentzAngleMap, stripUnit);
}

SiStripGPULocalToGlobalMap::~SiStripGPULocalToGlobalMap() {
  if (localToGlobalMap_) {
    cudaCheck(cudaFreeHost(localToGlobalMap_));
  }
}

LocalToGlobalMap const* SiStripGPULocalToGlobalMap::getGPUProductAsync(cudaStream_t stream) const {
  auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](GPUData& data, cudaStream_t stream) {
    // Allocate the payload object on the device memory.
    cudaCheck(cudaMalloc(&data.localToGlobalMapDevice_, sizeof(LocalToGlobalMap)));
    cudaCheck(cudaMemcpyAsync(
        data.localToGlobalMapDevice_, localToGlobalMap_, sizeof(LocalToGlobalMap), cudaMemcpyDefault, stream));
  });
  // Returns the payload object on the memory of the current device
  return data.localToGlobalMapDevice_;
}

SiStripGPULocalToGlobalMap::GPUData::~GPUData() { cudaCheck(cudaFree(localToGlobalMapDevice_)); }

void SiStripGPULocalToGlobalMap::loadBarrel(const std::vector<const GeometricDet*>& dets_barrel,
                                            const SiStripBackPlaneCorrection& BackPlaneCorrectionMap,
                                            const MagneticField& MagFieldMap,
                                            const SiStripLorentzAngle& LorentzAngleMap,
                                            const stripunit_v& stripUnit) {
  auto indexer3_h = localToGlobalMap_->indexer3;
  auto indexer5_h = localToGlobalMap_->indexer5;

  std::memset(indexer3_h, -1, sizeof(LocalToGlobalMap::indexer3));
  std::memset(indexer5_h, -1, sizeof(LocalToGlobalMap::indexer5));

  int* det_num_h = localToGlobalMap_->det_num_bd;
  double* pitch_h = localToGlobalMap_->pitch_bd;
  double* offset_h = localToGlobalMap_->offset_bd;
  double* len_h = localToGlobalMap_->len_bd;
  double* pos_x_h = localToGlobalMap_->pos_x_bd;
  double* pos_y_h = localToGlobalMap_->pos_y_bd;
  double* pos_z_h = localToGlobalMap_->pos_z_bd;
  double* R11_h = localToGlobalMap_->R11_bd;
  double* R12_h = localToGlobalMap_->R12_bd;
  double* R13_h = localToGlobalMap_->R13_bd;
  double* R21_h = localToGlobalMap_->R21_bd;
  double* R22_h = localToGlobalMap_->R22_bd;
  double* R23_h = localToGlobalMap_->R23_bd;
  double* backPlane_h = localToGlobalMap_->backPlane_bd;
  double* thickness_h = localToGlobalMap_->thickness_bd;
  double* drift_x_h = localToGlobalMap_->drift_x_bd;

  for (auto it = dets_barrel.begin(); it != dets_barrel.end(); ++it) {
    int i = std::distance(dets_barrel.begin(), it);
    const auto det = dets_barrel[i];
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
    backPlane_h[i] = BackPlaneCorrectionMap.getBackPlaneCorrection(rot_num);
    double lorentzAngle = LorentzAngleMap.getLorentzAngle(rot_num);
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

void SiStripGPULocalToGlobalMap::loadEndcap(const std::vector<const GeometricDet*>& dets_endcap,
                                            const SiStripBackPlaneCorrection& BackPlaneCorrectionMap,
                                            const MagneticField& MagFieldMap,
                                            const SiStripLorentzAngle& LorentzAngleMap,
                                            const stripunit_v& stripUnit) {
  short* indexer4_h = localToGlobalMap_->indexer4;
  short* indexer6_h = localToGlobalMap_->indexer6;
  cudaMemset(indexer4_h, -1, sizeof(LocalToGlobalMap::indexer4));
  cudaMemset(indexer6_h, -1, sizeof(LocalToGlobalMap::indexer6));

  int* det_num_h = localToGlobalMap_->det_num_ed;
  int* yAx_h = localToGlobalMap_->yAx_ed;
  double* backPlane_h = localToGlobalMap_->backPlane_ed;
  double* thickness_h = localToGlobalMap_->thickness_ed;
  double* drift_x_h = localToGlobalMap_->drift_x_ed;
  double* drift_y_h = localToGlobalMap_->drift_y_ed;
  double* rCross_h = localToGlobalMap_->rCross_ed;
  double* aw_h = localToGlobalMap_->aw_ed;
  double* phi_h = localToGlobalMap_->phi_ed;
  double* len_h = localToGlobalMap_->len_ed;
  double* pos_x_h = localToGlobalMap_->pos_x_ed;
  double* pos_y_h = localToGlobalMap_->pos_y_ed;
  double* pos_z_h = localToGlobalMap_->pos_z_ed;
  double* R11_h = localToGlobalMap_->R11_ed;
  double* R12_h = localToGlobalMap_->R12_ed;
  double* R13_h = localToGlobalMap_->R13_ed;
  double* R21_h = localToGlobalMap_->R21_ed;
  double* R22_h = localToGlobalMap_->R22_ed;
  double* R23_h = localToGlobalMap_->R23_ed;

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
    backPlane_h[i] = BackPlaneCorrectionMap.getBackPlaneCorrection(rot_num);
    double lorentzAngle = LorentzAngleMap.getLorentzAngle(rot_num);
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
