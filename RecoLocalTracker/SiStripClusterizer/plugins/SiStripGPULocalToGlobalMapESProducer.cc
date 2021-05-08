/**\class SiStripGPULocalToGlobalMapESProducer
 *
 * Create a GPU object for fast access to geometry needed by the SiStrip local to global conversion
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
};

SiStripGPULocalToGlobalMapESProducer::SiStripGPULocalToGlobalMapESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("Label"));

  m_geomDetToken = cc.consumesFrom<GeometricDet, IdealGeometryRecord>();
  m_magFieldToken = cc.consumesFrom<MagneticField, IdealMagneticFieldRecord>();
  m_backplaneToken = cc.consumesFrom<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionDepRcd>();
  m_lorentzAngleToken =
      cc.consumesFrom<SiStripLorentzAngle, SiStripLorentzAngleRcd>(edm::ESInputTag{"", "deconvolution"});
  m_trackerGeometryToken = cc.consumesFrom<TrackerGeometry, TrackerDigiGeometryRecord>(
      edm::ESInputTag{"", iConfig.getParameter<std::string>("QualityLabel")});
}

void SiStripGPULocalToGlobalMapESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("QualityLabel", "");
  desc.add<std::string>("Label", "");
  descriptions.add("SiStripGPULocalToGlobalMapESProducer", desc);
}

SiStripGPULocalToGlobalMapESProducer::ReturnType SiStripGPULocalToGlobalMapESProducer::produce(
    const SiStripGPULocalToGlobalMapRcd& iRecord) {
  return std::make_unique<SiStripGPULocalToGlobalMap>(iRecord.get(m_geomDetToken),
                                                      iRecord.get(m_magFieldToken),
                                                      iRecord.get(m_backplaneToken),
                                                      iRecord.get(m_lorentzAngleToken),
                                                      iRecord.get(m_trackerGeometryToken));
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripGPULocalToGlobalMapESProducer);
