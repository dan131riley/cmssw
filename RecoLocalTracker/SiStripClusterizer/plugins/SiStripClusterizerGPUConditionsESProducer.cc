/**\class SiStripClusterizerGPUConditionsESProducer
 *
 * Create a GPU cache object for fast access to conditions needed by the SiStrip clusterizer
 *
 * @see SiStripClusterizerConditions
 */
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoLocalTracker/Records/interface/SiStripClusterizerGPUConditionsRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "SiStripConditionsGPUWrapper.h"

class SiStripClusterizerGPUConditionsESProducer : public edm::ESProducer {
public:
  SiStripClusterizerGPUConditionsESProducer(const edm::ParameterSet&);
  ~SiStripClusterizerGPUConditionsESProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using ReturnType = std::unique_ptr<SiStripConditionsGPUWrapper>;
  ReturnType produce(const SiStripClusterizerGPUConditionsRcd&);

private:
  edm::ESGetToken<SiStripGain, SiStripGainRcd> m_gainToken;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> m_noisesToken;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> m_qualityToken;
};

SiStripClusterizerGPUConditionsESProducer::SiStripClusterizerGPUConditionsESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("Label"));

  m_gainToken = cc.consumesFrom<SiStripGain, SiStripGainRcd>();
  m_noisesToken = cc.consumesFrom<SiStripNoises, SiStripNoisesRcd>();
  m_qualityToken = cc.consumesFrom<SiStripQuality, SiStripQualityRcd>(
      edm::ESInputTag{"", iConfig.getParameter<std::string>("QualityLabel")});
}

void SiStripClusterizerGPUConditionsESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("QualityLabel", "");
  desc.add<std::string>("Label", "");
  descriptions.add("SiStripClusterizerGPUConditionsESProducer", desc);
}

SiStripClusterizerGPUConditionsESProducer::ReturnType SiStripClusterizerGPUConditionsESProducer::produce(
    const SiStripClusterizerGPUConditionsRcd& iRecord) {
  auto gainsH = iRecord.getTransientHandle(m_gainToken);
  const auto& noises = iRecord.get(m_noisesToken);
  const auto& quality = iRecord.get(m_qualityToken);

  return std::make_unique<SiStripConditionsGPUWrapper>(quality, gainsH.product(), noises);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripClusterizerGPUConditionsESProducer);
