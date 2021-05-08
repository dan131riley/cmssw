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

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitStripInputWrapper.h"
#include "Hit.h"
#include "LayerNumberConverter.h"

#include "RecoLocalTracker/SiStripClusterizer/plugins/MkFitSiStripHitGPUKernel.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitRecHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoLocalTracker/Records/interface/SiStripGPULocalToGlobalMapRcd.h"
#include "SiStripGPULocalToGlobalMap.h"

class MkFitSiStripHitsFromSOA final : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit MkFitSiStripHitsFromSOA(const edm::ParameterSet& conf) {
    inputToken_ = consumes<cms::cuda::Product<SiStripClustersCUDA>>(conf.getParameter<edm::InputTag>("siClusters"));
    pixelhitToken_ = consumes<MkFitHitWrapper>(conf.getParameter<edm::InputTag>("pixelhits"));

    geometryToken_ = esConsumes<SiStripGPULocalToGlobalMap, SiStripGPULocalToGlobalMapRcd>(
        edm::ESInputTag{"", conf.getParameter<std::string>("ConditionsLabel")});
    topologyToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();

    outputToken_ = produces<MkFitRecHitWrapper>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginRun(const edm::Run&, const edm::EventSetup& es) override {}

  void acquire(edm::Event const& ev,
               edm::EventSetup const& es,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override {
    const auto& wrapper = ev.get(inputToken_);
    const auto& geometry = es.getData(geometryToken_);

    // Sets the current device and creates a CUDA stream
    cms::cuda::ScopedContextAcquire ctx{wrapper, std::move(waitingTaskHolder)};

    const auto& input = ctx.get(wrapper);
    // Queues asynchronous data transfers and kernels to the CUDA stream
    // returned by cms::cuda::ScopedContextAcquire::stream()
    gpuAlgo_.makeGlobal(input, geometry.getGPUProductAsync(ctx.stream()), clusters_g, ctx.stream());
    hostView_x = clusters_g.hostView(SiStripClustersCUDA::kClusterMaxStrips, ctx.stream());

    // Destructor of ctx queues a callback to the CUDA stream notifying
    // waitingTaskHolder when the queued asynchronous work has finished
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    int totalHits = ev.get(pixelhitToken_).totalHits();

    mkfit::LayerNumberConverter lnc{mkfit::TkLayout::phase1};

    std::unique_ptr<MkFitSiStripClustersCUDA::HostView> clust_data = std::move(hostView_x);
    const int nSeedStripsNC = clust_data->nClusters_h;
    const auto global_x = clust_data->global_x_h.get();
    const auto global_y = clust_data->global_y_h.get();
    const auto global_z = clust_data->global_z_h.get();
    const auto global_xx = clust_data->global_xx_h.get();
    const auto global_xy = clust_data->global_xy_h.get();
    const auto global_xz = clust_data->global_xz_h.get();
    const auto global_yy = clust_data->global_yy_h.get();
    const auto global_yz = clust_data->global_yz_h.get();
    const auto global_zz = clust_data->global_zz_h.get();
    const auto layer = clust_data->layer_h.get();
    const auto detid = clust_data->clusterDetId_h.get();
    const auto barycenter = clust_data->barycenter_h.get();

    std::vector<std::vector<float>> set_barycenters(lnc.nLayers());
    std::vector<std::vector<int>> set_detIds(lnc.nLayers());
    std::vector<mkfit::HitVec> mkFitHits(lnc.nLayers());

    for (int j = 0; j < static_cast<int>(lnc.nLayers()); j++) {
      mkFitHits[j].reserve(5000);
      set_barycenters[j].reserve(5000);
      set_detIds[j].reserve(5000);
    }

    const auto& ttopo = es.getData(topologyToken_);
    using SVector3 = ROOT::Math::SVector<float, 3>;
    using SMatrixSym33 = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;
    std::vector<uint8_t> adcs;
    adcs.reserve(SiStripClustersCUDA::kClusterMaxStrips);
    for (int i = 0; i < nSeedStripsNC; ++i) {
      if (layer[i] == -1) {
        continue;
      }  // layer number doubles as "bad hit" index
      SVector3 pos(global_x[i], global_y[i], global_z[i]);
      SMatrixSym33 err;
      err.At(0, 0) = global_xx[i];
      err.At(0, 1) = global_xy[i];
      err.At(0, 2) = global_xz[i];
      err.At(1, 1) = global_yy[i];
      err.At(1, 2) = global_yz[i];
      err.At(2, 2) = global_zz[i];
      int subdet = (detid[i] >> 25) & 0x7;
      bool stereoraw = ttopo.isStereo(detid[i]);
      bool plusraw = (ttopo.side(detid[i]) == static_cast<unsigned>(TrackerDetSide::PosEndcap));
      const auto ilay = lnc.convertLayerNumber(subdet, layer[i], false, stereoraw, plusraw);
      mkFitHits[ilay].emplace_back(pos, err, totalHits);
      set_barycenters[ilay].emplace_back(barycenter[i]);
      set_detIds[ilay].emplace_back(detid[i]);
      ++totalHits;
    }

    mkFitHits.shrink_to_fit();
    set_barycenters.shrink_to_fit();
    set_detIds.shrink_to_fit();

    ev.emplace(outputToken_, std::move(mkFitHits), totalHits, std::move(set_barycenters), std::move(set_detIds));
  }

private:
  stripgpu::MkFitSiStripHitGPUKernel gpuAlgo_;
  MkFitSiStripClustersCUDA clusters_g;
  std::unique_ptr<MkFitSiStripClustersCUDA::HostView> hostView_x;

  edm::EDGetTokenT<cms::cuda::Product<SiStripClustersCUDA>> inputToken_;
  edm::EDPutTokenT<MkFitRecHitWrapper> outputToken_;
  edm::EDGetTokenT<MkFitHitWrapper> pixelhitToken_;

  edm::ESGetToken<SiStripGPULocalToGlobalMap, SiStripGPULocalToGlobalMapRcd> geometryToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topologyToken_;
};

void MkFitSiStripHitsFromSOA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add("pixelhits", edm::InputTag{"mkFitPixelConverter"});
  desc.add("siClusters", edm::InputTag{"SiStripClustersFromRawFacility"});
  desc.add("stripRphiRecHits", edm::InputTag{"siStripMatchedRecHits", "rphiRecHit"});
  desc.add("stripStereoRecHits", edm::InputTag{"siStripMatchedRecHits", "stereoRecHit"});
  desc.add<std::string>("ConditionsLabel", "");
  descriptions.add("mkFitHitsFromSOADefault", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MkFitSiStripHitsFromSOA);
