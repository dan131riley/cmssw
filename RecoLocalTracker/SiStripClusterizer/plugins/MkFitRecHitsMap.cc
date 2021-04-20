/*
 */
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "CUDADataFormats/SiStripCluster/interface/MkFitSiStripClustersCUDA.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include <memory>

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitStripInputWrapper.h"
#include "Hit.h"
#include "LayerNumberConverter.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "RecoLocalTracker/SiStripClusterizer/plugins/MkFitSiStripHitGPUKernel.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitRecHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
class MkFitSiStripRecHitsMap final : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit MkFitSiStripRecHitsMap(const edm::ParameterSet& conf) {
    stripRphiRecHitToken_= consumes<SiStripRecHit2DCollection>(conf.getParameter<edm::InputTag>("stripRphiRecHits"));
    stripStereoRecHitToken_ = consumes<SiStripRecHit2DCollection>(conf.getParameter<edm::InputTag>("stripStereoRecHits"));
    outputToken_ = produces<MkFitHitWrapper>();
    pixelhitToken_ = consumes<MkFitHitWrapper>(conf.getParameter<edm::InputTag>("pixelhits"));
    striphitToken_ = consumes<MkFitRecHitWrapper>(conf.getParameter<edm::InputTag>("striphits"));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
template <typename HitCollection>
bool fillMap(const HitCollection& hits, unsigned int detid, float barycenter,int size,int ilay, MkFitHitIndexMap& hitIndexMap);

  void beginRun(const edm::Run&, const edm::EventSetup& es) override {
  }

  void acquire(edm::Event const& ev,
               edm::EventSetup const& es,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override {
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    MkFitHitIndexMap hitIndexMap = ev.get(pixelhitToken_).hitIndexMap();
    std::vector<mkfit::HitVec> mkFitHits = ev.get(pixelhitToken_).hits();
    std::vector<mkfit::HitVec> strip_mkFitHits = ev.get(striphitToken_).hits();
    std::vector<std::vector<float>> barycenters = ev.get(striphitToken_).barycenters();
    std::vector<std::vector<int>> detIds = ev.get(striphitToken_).detIds();


    for( int ilay=0; ilay < static_cast<int>(strip_mkFitHits.size()); ilay++){
      int index = 0;
      for(int i =0; i<static_cast<int>(strip_mkFitHits[ilay].size());i++){
        bool found_hit;
        int detid = detIds[ilay][i];
        float barycenter = barycenters[ilay][i];
        found_hit = fillMap(ev.get(stripRphiRecHitToken_),detid,barycenter,index,ilay,hitIndexMap);
        if(!found_hit){
          found_hit = fillMap(ev.get(stripStereoRecHitToken_),detid,barycenter,index,ilay,hitIndexMap);
        }
        if(found_hit){ 
          mkFitHits[ilay].push_back(strip_mkFitHits[ilay][i]);
          index++;
        }
      }
    }
    ev.emplace(outputToken_,std::move(hitIndexMap),std::move(mkFitHits),0);
  }

private:
  stripgpu::MkFitSiStripHitGPUKernel gpuAlgo_;
  MkFitSiStripClustersCUDA clusters_g;
  std::unique_ptr<MkFitSiStripClustersCUDA::HostView> hostView_x;

  edm::EDPutTokenT<MkFitHitWrapper> outputToken_;
  edm::EDGetTokenT<MkFitHitWrapper> pixelhitToken_;
  edm::EDGetTokenT<MkFitRecHitWrapper> striphitToken_;


  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  SiStripRecHit2DCollection rechits_stereo;
  SiStripRecHit2DCollection rechits_phi;
  edmNew::DetSetVector<SiStripRecHit2DCollection> rechits;
const TrackerGeometry* tkG;
};


template <typename HitCollection>
inline bool MkFitSiStripRecHitsMap::fillMap(const HitCollection& hits, unsigned int detid, float barycenter, int size,int ilay, MkFitHitIndexMap& hitIndexMapx){
      bool pass = false;
      float bary_epsilon = 2;
      for (const auto& detset: hits){
        if(pass){break;}
        const DetId detid_clust = detset.detId();
        if(detid_clust.rawId() != detid) {continue;}
        hitIndexMapx.increaseLayerSize(ilay, detset.size());
        for (const auto& hit : detset) {
          if(pass){break;}
          auto bary = hit.cluster()->barycenter();
          if (abs(bary - barycenter) > bary_epsilon) {continue;}
          hitIndexMapx.insert(hit.firstClusterRef().id(),
                         hit.firstClusterRef().index(),
                         MkFitHitIndexMap::MkFitHit{size, ilay},
                         &hit);
          pass = true;
          break;
        }
      }
      return pass;
}
void MkFitSiStripRecHitsMap::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("pixelhits", edm::InputTag{"mkFitPixelConverter"});
  desc.add("striphits", edm::InputTag{"mkFitStripConverter"});
  desc.add("siClusters", edm::InputTag{"SiStripClustersFromRawFacility"});
  desc.add("stripRphiRecHits", edm::InputTag{"siStripMatchedRecHits", "rphiRecHit"});
  desc.add("stripStereoRecHits", edm::InputTag{"siStripMatchedRecHits", "stereoRecHit"});
  descriptions.add("mkFitRecHitsMapDefault", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MkFitSiStripRecHitsMap);
