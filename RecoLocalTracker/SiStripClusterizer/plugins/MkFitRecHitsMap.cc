/*
 */
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CUDADataFormats/SiStripCluster/interface/MkFitSiStripClustersCUDA.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include <unordered_map>
#include <memory>
#include <atomic>
#include <set>

#include "Hit.h"
#include "LayerNumberConverter.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitStripInputWrapper.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitRecHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

class MkFitSiStripRecHitsMap final : public edm::stream::EDProducer<> {
public:
  explicit MkFitSiStripRecHitsMap(const edm::ParameterSet& conf) {
    stripRphiRecHitToken_ = consumes<SiStripRecHit2DCollection>(conf.getParameter<edm::InputTag>("stripRphiRecHits"));
    stripStereoRecHitToken_ =
        consumes<SiStripRecHit2DCollection>(conf.getParameter<edm::InputTag>("stripStereoRecHits"));
    pixelhitToken_ = consumes<MkFitHitWrapper>(conf.getParameter<edm::InputTag>("pixelhits"));
    striphitToken_ = consumes<MkFitRecHitWrapper>(conf.getParameter<edm::InputTag>("striphits"));
    outputToken_ = produces<MkFitHitWrapper>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void fillDetSetMap(const SiStripRecHit2DCollection& hits);

  bool fillMap(unsigned int detid, int firstStrip, int index, int ilay, MkFitHitIndexMap& hitIndexMap);

  void beginRun(const edm::Run&, const edm::EventSetup& es) override {}

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    auto&& strip_mkFitHits = ev.get(striphitToken_).hits();
    auto&& detIds = ev.get(striphitToken_).detIds();
    auto&& firstStrips = ev.get(striphitToken_).firstStrips();
    // these get modified by adding the strip hits, so we copy
    auto hitIndexMap = ev.get(pixelhitToken_).hitIndexMap();
    auto mkFitHits = ev.get(pixelhitToken_).hits();

    detSetMap_.clear();

    fillDetSetMap(ev.get(stripRphiRecHitToken_));
    fillDetSetMap(ev.get(stripStereoRecHitToken_));

    for (int ilay = 0; ilay < static_cast<int>(strip_mkFitHits.size()); ilay++) {
      int index = 0;
      hitIndexMap.increaseLayerSize(ilay, strip_mkFitHits[ilay].size());
      for (int i = 0; i < static_cast<int>(strip_mkFitHits[ilay].size()); i++) {
        const auto detid = detIds[ilay][i];
        const auto firstStrip = firstStrips[ilay][i];
        if (fillMap(detid, firstStrip, index, ilay, hitIndexMap)) {
          mkFitHits[ilay].push_back(strip_mkFitHits[ilay][i]);
          ++index;
        }
      }
    }
    ev.emplace(outputToken_, std::move(hitIndexMap), std::move(mkFitHits), ev.get(striphitToken_).totalHits());
  }

private:
  using HitCollection = SiStripRecHit2DCollection::value_type;

  std::unordered_map<DetId, const HitCollection> detSetMap_;

  edm::EDPutTokenT<MkFitHitWrapper> outputToken_;

  edm::EDGetTokenT<MkFitHitWrapper> pixelhitToken_;
  edm::EDGetTokenT<MkFitRecHitWrapper> striphitToken_;

  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
};

void MkFitSiStripRecHitsMap::fillDetSetMap(const SiStripRecHit2DCollection& hits) {
  for (const auto& detset : hits) {
    detSetMap_.emplace(detset.detId(), detset);
  }
}

bool MkFitSiStripRecHitsMap::fillMap(
    unsigned int detid, int firstStrip, int index, int ilay, MkFitHitIndexMap& hitIndexMapx) {
  const auto detiter = detSetMap_.find(detid);
  if (detiter != detSetMap_.end()) {
    const auto& detset{detiter->second};
    for (const auto& hit : detset) {
      if (firstStrip == hit.cluster()->firstStrip()) {
        hitIndexMapx.insert(
            hit.firstClusterRef().id(), hit.firstClusterRef().index(), MkFitHitIndexMap::MkFitHit{index, ilay}, &hit);
        return true;
      }
    }
  }
  return false;
}
void MkFitSiStripRecHitsMap::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("pixelhits", edm::InputTag{"mkFitPixelConverter"});
  desc.add("striphits", edm::InputTag{"mkFitStripConverter"});
  desc.add("stripRphiRecHits", edm::InputTag{"siStripMatchedRecHits", "rphiRecHit"});
  desc.add("stripStereoRecHits", edm::InputTag{"siStripMatchedRecHits", "stereoRecHit"});
  descriptions.add("mkFitRecHitsMapDefault", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MkFitSiStripRecHitsMap);
