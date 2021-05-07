/**\class SiStripGPULocalToGlobalMapESProducer
 *
 * Create a GPU cache object for fast access to geometry needed by the SiStrip local to global conversion
 *
 */
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoLocalTracker/Records/interface/SiStripGPULocalToGlobalMapRcd.h"

#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "SiStripGPULocalToGlobalMap.h"

class SiStripGPULocalToGlobalMapESProducer : public edm::ESProducer {
public:
  SiStripGPULocalToGlobalMapESProducer(const edm::ParameterSet&);
  ~SiStripGPULocalToGlobalMapESProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using ReturnType = std::unique_ptr<SiStripGPULocalToGlobalMap>;
  ReturnType produce(const SiStripGPULocalToGlobalMapRcd&);

private:
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> m_geomDetToken;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_magFieldToken;
  edm::ESGetToken<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionDepRcd> m_backplaneToken;
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> m_lorentzAngleToken;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> m_trackerGeometryToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_trackerTopologyToken;
};

SiStripGPULocalToGlobalMapESProducer::SiStripGPULocalToGlobalMapESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("Label"));

  m_geomDetToken = cc.consumesFrom<GeometricDet, IdealGeometryRecord>();
  m_magFieldToken = cc.consumesFrom<MagneticField, IdealMagneticFieldRecord>();
  m_backplaneToken = cc.consumesFrom<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionDepRcd>();
  m_lorentzAngleToken =
      cc.consumesFrom<SiStripLorentzAngle, SiStripLorentzAngleRcd>(edm::ESInputTag{"", "deconvolution"});
  m_trackerGeometryToken = cc.consumesFrom<TrackerGeometry, TrackerDigiGeometryRecord>();
  m_trackerTopologyToken = cc.consumesFrom<TrackerTopology, TrackerTopologyRcd>();
}

void SiStripGPULocalToGlobalMapESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("SiStripGPULocalToGlobalMapESProducer", desc);
}

SiStripGPULocalToGlobalMapESProducer::ReturnType SiStripGPULocalToGlobalMapESProducer::produce(
    const SiStripGPULocalToGlobalMapRcd& iRecord) {
  const auto& GeomDet2 = iRecord.get(m_geomDetToken);
  const auto& magField = iRecord.get(m_magFieldToken);
  const auto& backPlane = iRecord.get(m_backplaneToken);
  const auto& lorentz = iRecord.get(m_lorentzAngleToken);
  const auto& tkGx = iRecord.get(m_trackerGeometryToken);

  return std::make_unique<SiStripGPULocalToGlobalMap>(GeomDet2, magField, backPlane, lorentz, tkGx);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripGPULocalToGlobalMapESProducer);
