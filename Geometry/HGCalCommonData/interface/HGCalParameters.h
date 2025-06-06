#ifndef Geometry_HGCalCommonData_HGCalParameters_h
#define Geometry_HGCalCommonData_HGCalParameters_h

#include <CLHEP/Geometry/Transform3D.h>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "DD4hep/DD4hepUnits.h"

class HGCalParameters {
public:
  struct waferInfo {
    int32_t type, part, orient, cassette;
    waferInfo(int32_t t = 0, int32_t p = 0, int32_t o = 0, int32_t c = 0) : type(t), part(p), orient(o), cassette(c) {}
  };
  struct tileInfo {
    int32_t type, sipm, cassette, hex[6];
    tileInfo(int32_t t = 0,
             int32_t s = 0,
             int32_t h1 = 0,
             int32_t h2 = 0,
             int32_t h3 = 0,
             int32_t h4 = 0,
             int32_t h5 = 0,
             int32_t h6 = 0)
        : type(t), sipm(s) {
      hex[0] = h1;
      hex[1] = h2;
      hex[2] = h3;
      hex[3] = h4;
      hex[4] = h5;
      hex[5] = h6;
    };
  };
  typedef std::vector<std::unordered_map<int32_t, int32_t> > layer_map;
  typedef std::unordered_map<int32_t, int32_t> wafer_map;
  typedef std::unordered_map<int32_t, std::pair<int32_t, int32_t> > waferT_map;
  typedef std::unordered_map<int32_t, waferInfo> waferInfo_map;
  typedef std::unordered_map<int32_t, tileInfo> tileInfo_map;

  static constexpr double k_ScaleFromDDD = 0.1;
  static constexpr double k_ScaleFromDDD2 = 0.01;
  static constexpr double k_ScaleToDDD = 10.0;
  static constexpr double k_ScaleToDDD2 = 100.0;
  static constexpr double k_ScaleFromDDDToG4 = 1.0;
  static constexpr double k_ScaleToDDDFromG4 = 1.0;
  static constexpr double k_ScaleFromDD4hep = (1.0 / dd4hep::cm);
  static constexpr double k_ScaleToDD4hep = dd4hep::cm;
  static constexpr double k_ScaleFromDD4hepToG4 = (1.0 / dd4hep::mm);
  static constexpr double k_ScaleToDD4hepFromG4 = dd4hep::mm;
  static constexpr uint32_t k_CornerSize = 6;
  static constexpr double tol = 1.0e-12;

  struct hgtrap {
    int lay;
    float bl, tl, h, dz, alpha, cellSize;
  };

  struct hgtrform {
    int zp, lay, sec, subsec;
    CLHEP::Hep3Vector h3v;
    CLHEP::HepRotation hr;
  };

  HGCalParameters(const std::string& nam);
  ~HGCalParameters(void) = default;
  void fillModule(const hgtrap& mytr, bool reco);
  hgtrap getModule(unsigned int k, bool reco) const;
  void fillTrForm(const hgtrform& mytr);
  hgtrform getTrForm(unsigned int k) const;
  void addTrForm(const CLHEP::Hep3Vector& h3v);
  void scaleTrForm(double);
  int scintCells(const int layer) const { return nPhiBinBH_[scintType(layer)]; }
  double scintCellSize(const int layer) const { return cellSize_[scintType(layer)]; }
  bool scintFine(int indx) const { return ((!tileRingFineR_.empty()) && (nPhiLayer_[indx] > 288)); }
  double scintRing(int indx, int irad) const;
  int scintType(const int layer) const { return ((layer < layerFrontBH_[1]) ? 1 : 0); }
  bool scintValidRing(int indx, int irad) const {
    return (scintFine(indx) ? ((irad >= iradMinBHFine_[indx]) && (irad <= (iradMaxBHFine_[indx] + 1)))
                            : ((irad >= iradMinBH_[indx]) && (irad <= (iradMaxBH_[indx] + 1))));
  }
  std::array<int, 4> getID(unsigned int k) const;

  std::string name_;
  int detectorType_ = 0;
  int useSimWt_ = 0;
  int nCells_ = 0;
  int nSectors_ = 0;
  int firstLayer_ = 0;
  int firstMixedLayer_ = 0;
  HGCalGeometryMode::GeometryMode mode_ = HGCalGeometryMode::Square;

  std::vector<double> cellSize_;
  std::vector<double> slopeMin_;
  std::vector<double> zFrontMin_;
  std::vector<double> rMinFront_;
  std::vector<double> slopeTop_;
  std::vector<double> zFrontTop_;
  std::vector<double> rMaxFront_;
  std::vector<double> zRanges_;
  std::vector<int> moduleLayS_;
  std::vector<double> moduleBlS_;
  std::vector<double> moduleTlS_;
  std::vector<double> moduleHS_;
  std::vector<double> moduleDzS_;
  std::vector<double> moduleAlphaS_;
  std::vector<double> moduleCellS_;
  std::vector<int> moduleLayR_;
  std::vector<double> moduleBlR_;
  std::vector<double> moduleTlR_;
  std::vector<double> moduleHR_;
  std::vector<double> moduleDzR_;
  std::vector<double> moduleAlphaR_;
  std::vector<double> moduleCellR_;
  std::vector<uint32_t> trformIndex_;
  std::vector<double> trformTranX_;
  std::vector<double> trformTranY_;
  std::vector<double> trformTranZ_;
  std::vector<double> trformRotXX_;
  std::vector<double> trformRotYX_;
  std::vector<double> trformRotZX_;
  std::vector<double> trformRotXY_;
  std::vector<double> trformRotYY_;
  std::vector<double> trformRotZY_;
  std::vector<double> trformRotXZ_;
  std::vector<double> trformRotYZ_;
  std::vector<double> trformRotZZ_;
  std::vector<double> xLayerHex_;
  std::vector<double> yLayerHex_;
  std::vector<double> zLayerHex_;
  std::vector<double> rMinLayHex_;
  std::vector<double> rMaxLayHex_;
  std::vector<double> waferPosX_;
  std::vector<double> waferPosY_;
  wafer_map cellFineIndex_;
  std::vector<double> cellFineX_;
  std::vector<double> cellFineY_;
  std::vector<bool> cellFineHalf_;
  wafer_map cellCoarseIndex_;
  std::vector<double> cellCoarseX_;
  std::vector<double> cellCoarseY_;
  std::vector<bool> cellCoarseHalf_;
  std::vector<double> boundR_;
  std::vector<int> layer_;
  std::vector<int> layerIndex_;
  std::vector<int> layerGroup_;
  std::vector<int> cellFactor_;
  std::vector<int> depth_;
  std::vector<int> depthIndex_;
  std::vector<int> depthLayerF_;
  std::vector<int> waferCopy_;
  std::vector<int> waferTypeL_;
  std::vector<int> waferTypeT_;
  std::vector<int> layerGroupM_;
  std::vector<int> layerGroupO_;
  std::vector<double> rLimit_;
  std::vector<int> cellFine_;
  std::vector<int> cellCoarse_;
  double waferR_ = 0.;
  std::vector<int> levelT_;
  int levelZSide_ = 0;
  layer_map copiesInLayers_;
  int nCellsFine_ = 0;
  int nCellsCoarse_ = 0;
  double waferSize_ = 0.;
  double waferSizeNominal_ = 0.;
  double waferThick_ = 0.;
  double sensorSeparation_ = 0.;
  double sensorSizeOffset_ = 0.;
  double guardRingOffset_ = 0.;
  double mouseBite_ = 0.;
  int useOffset_ = 0;
  int waferUVMax_ = 0;
  std::vector<int> waferUVMaxLayer_;
  bool defineFull_ = false;
  std::vector<double> waferThickness_;
  std::vector<double> cellThickness_;
  std::vector<double> radius100to200_;
  std::vector<double> radius200to300_;
  int choiceType_ = 0;
  int nCornerCut_ = 0;
  double fracAreaMin_ = 0.;
  double zMinForRad_ = 0.;
  std::vector<double> radiusMixBoundary_;
  std::vector<int> nPhiBinBH_;
  std::vector<int> layerFrontBH_;
  std::vector<double> rMinLayerBH_;
  std::vector<double> radiusLayer_[2];
  std::vector<int> iradMinBH_;
  std::vector<int> iradMaxBH_;
  std::vector<int> iradMinBHFine_;
  std::vector<int> iradMaxBHFine_;
  double minTileSize_ = 0.;
  std::vector<int> firstModule_;
  std::vector<int> lastModule_;
  int layerOffset_ = 0;
  double layerRotation_ = 0.;
  std::vector<int> layerType_;
  std::vector<int> layerCenter_;
  wafer_map wafersInLayers_;
  wafer_map typesInLayers_;
  waferT_map waferTypes_;
  int waferMaskMode_ = 0;
  int waferNoGap_ = 0;
  int waferZSide_ = 0;
  waferInfo_map waferInfoMap_;
  std::vector<std::pair<double, double> > layerRotV_;
  tileInfo_map tileInfoMap_;
  std::vector<std::pair<double, double> > tileRingR_;
  std::vector<std::pair<int, int> > tileRingRange_;
  std::vector<std::pair<double, double> > tileRingFineR_;
  std::vector<std::pair<int, int> > tileRingFineRange_;
  std::vector<int> nPhiLayer_;
  int cassettes_ = 0;
  int nphiCassette_ = 0;
  int nphiFineCassette_ = 0;
  int phiOffset_ = 0;
  std::vector<double> cassetteShift_;
  std::vector<double> cassetteShiftTile_;
  std::vector<double> cassetteRetractTile_;
  double calibCellRHD_ = 0.;
  std::vector<int> calibCellFullHD_;
  std::vector<int> calibCellPartHD_;
  double calibCellRLD_ = 0.;
  std::vector<int> calibCellFullLD_;
  std::vector<int> calibCellPartLD_;
  int tileUVMax_ = 0;
  int tileUVMaxFine_ = 0;

  COND_SERIALIZABLE;

private:
  static constexpr int kMaskZside = 0x1;
  static constexpr int kMaskLayer = 0x7F;
  static constexpr int kMaskSector = 0x3FF;
  static constexpr int kMaskSubSec = 0x1;
  static constexpr int kShiftZside = 19;
  static constexpr int kShiftLayer = 12;
  static constexpr int kShiftSector = 1;
  static constexpr int kShiftSubSec = 0;
};

#endif
