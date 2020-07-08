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

#include "clusterGPU.cuh"

#include <memory>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class SiStripSOAtoHost {
public:
  SiStripSOAtoHost() = default;
  void makeAsync(const SiStripClustersCUDA& clusters_d, cudaStream_t stream) {
    hostView_ = clusters_d.hostView(kClusterMaxStrips, stream);
  }
  std::unique_ptr<SiStripClustersCUDA::HostView> getResults() {
    return std::move(hostView_);
  }
private:
  std::unique_ptr<SiStripClustersCUDA::HostView> hostView_;
};

class SiStripClustersFromSOA final : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiStripClustersFromSOA(const edm::ParameterSet& conf) {
    inputToken_ = consumes<cms::cuda::Product<SiStripClustersCUDA>>(conf.getParameter<edm::InputTag>("ProductLabel"));
    outputToken_ = produces<edmNew::DetSetVector<SiStripCluster>>();
  }

  void beginRun(const edm::Run&, const edm::EventSetup& es) override { 
  }

  void acquire(edm::Event const& ev, edm::EventSetup const& es, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override {
    const auto& wrapper = ev.get(inputToken_);

    // Sets the current device and creates a CUDA stream
    cms::cuda::ScopedContextAcquire ctx{wrapper, std::move(waitingTaskHolder)};

    const auto& input = ctx.get(wrapper);

    // Queues asynchronous data transfers and kernels to the CUDA stream
    // returned by cms::cuda::ScopedContextAcquire::stream()
    gpuAlgo_.makeAsync(input, ctx.stream());

    // Destructor of ctx queues a callback to the CUDA stream notifying
    // waitingTaskHolder when the queued asynchronous work has finished
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    //cms::cuda::ScopedContextProduce ctx{ctxState_};

    using out_t = edmNew::DetSetVector<SiStripCluster>;
    std::unique_ptr<out_t> output(new edmNew::DetSetVector<SiStripCluster>());

    auto clust_data = gpuAlgo_.getResults();

    const int nSeedStripsNC = clust_data->nClusters_h;
    const auto clusterSize = clust_data->clusterSize_h.get();
    const auto ADCs = clust_data->clusterADCs_h.get();
    const auto detIDs = clust_data->clusterDetId_h.get();
    const auto stripIDs = clust_data->firstStrip_h.get();
    const auto trueCluster = clust_data->trueCluster_h.get();

    output->reserve(15000, nSeedStripsNC);

    std::vector<uint8_t> adcs;
    adcs.reserve(kClusterMaxStrips);

    for (int i = 0; i < nSeedStripsNC;) {
      const auto detid = detIDs[i];
      out_t::FastFiller record(*output, detid);

      while (i < nSeedStripsNC && detIDs[i] == detid) {
        if (trueCluster[i]) {
          const auto size = std::min(clusterSize[i], kClusterMaxStrips);
          const auto firstStrip = stripIDs[i];

          adcs.clear();
          for (uint32_t j = 0; j < size; ++j) {
            adcs.push_back(ADCs[i+j*nSeedStripsNC]);
          }
          record.push_back(SiStripCluster(firstStrip, adcs.begin(), adcs.end()));
        }
        i++;
      }

      //#define DSRDEBUG
#ifdef DSRDEBUG
      if (detid == 369120277) {
        std::cout << "Printing clusters for detid " << detid << std::endl;
        for (const auto& cluster : record) {
          std::cout << "Cluster " << cluster.firstStrip() << ": ";
          for (const auto& ampl : cluster.amplitudes()) {
            std::cout << (int) ampl << " ";
          }
          std::cout << std::endl;
        }
      }
#endif
    }

    output->shrink_to_fit();
    ev.put(std::move(output));
  }

private:
  SiStripSOAtoHost gpuAlgo_;

  edm::EDGetTokenT<cms::cuda::Product<SiStripClustersCUDA>> inputToken_;
  edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> outputToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClustersFromSOA);
