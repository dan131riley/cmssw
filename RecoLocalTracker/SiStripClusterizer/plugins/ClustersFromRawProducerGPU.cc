/*
 */
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "RecoLocalTracker/SiStripClusterizer/plugins/SiStripRawToClusterGPUKernel.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripConditionsGPUWrapper.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

//#include <sstream>
#include <memory>
#include <mutex>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

namespace {
  std::unique_ptr<sistrip::FEDBuffer> fillBuffer(int fedId, const FEDRawData& rawData) {
    std::unique_ptr<sistrip::FEDBuffer> buffer;

    // Check on FEDRawData pointer
    const auto st_buffer = sistrip::preconstructCheckFEDBuffer(rawData);
    if
      UNLIKELY(sistrip::FEDBufferStatusCode::SUCCESS != st_buffer) {
        if (edm::isDebugEnabled()) {
          edm::LogWarning(sistrip::mlRawToCluster_)
              << "[ClustersFromRawProducer::" << __func__ << "]" << st_buffer << " for FED ID " << fedId;
        }
        return buffer;
      }
    buffer = std::make_unique<sistrip::FEDBuffer>(rawData);
    const auto st_chan = buffer->findChannels();
    if
      UNLIKELY(sistrip::FEDBufferStatusCode::SUCCESS != st_chan) {
        if (edm::isDebugEnabled()) {
          edm::LogWarning(sistrip::mlRawToCluster_)
              << "Exception caught when creating FEDBuffer object for FED " << fedId << ": " << st_chan;
        }
        buffer.reset();
        return buffer;
      }
    if
      UNLIKELY(!buffer->doChecks(false)) {
        if (edm::isDebugEnabled()) {
          edm::LogWarning(sistrip::mlRawToCluster_)
              << "Exception caught when creating FEDBuffer object for FED " << fedId << ": FED Buffer check fails";
        }
        buffer.reset();
        return buffer;
      }

      return buffer;
  }
} // namespace

class SiStripClusterizerFromRawGPU final : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiStripClusterizerFromRawGPU(const edm::ParameterSet& conf)
      : buffers(1024),
        raw(1024),
        cabling_(nullptr),
        clusterizer_(StripClusterizerAlgorithmFactory::create(conf.getParameter<edm::ParameterSet>("Clusterizer"))),
        rawAlgos_(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
        doAPVEmulatorCheck_(conf.existsAs<bool>("DoAPVEmulatorCheck") ? conf.getParameter<bool>("DoAPVEmulatorCheck")
                                                                      : true),
        legacy_(conf.existsAs<bool>("LegacyUnpacker") ? conf.getParameter<bool>("LegacyUnpacker") : false),
        hybridZeroSuppressed_(conf.getParameter<bool>("HybridZeroSuppressed")) {
    productToken_ = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("ProductLabel"));
    produces<edmNew::DetSetVector<SiStripCluster> >();
    assert(clusterizer_.get());
    assert(rawAlgos_.get());
  }

  void beginRun(const edm::Run&, const edm::EventSetup& es) override { 
    initialize(es);
    conditionsWrapper = std::make_unique<SiStripConditionsGPUWrapper>(clusterizer_.get());
  }

  void acquire(edm::Event const& ev, edm::EventSetup const& es, edm::WaitingTaskWithArenaHolder waitingTaskHolder) override {
    //initialize(es); // ??

    // Sets the current device and creates a CUDA stream
    cms::cuda::ScopedContextAcquire ctx{ev.streamID(), std::move(waitingTaskHolder), ctxState_};

    // get raw data
    edm::Handle<FEDRawDataCollection> rawData;
    ev.getByToken(productToken_, rawData);

    run(*rawData);

    // Queues asynchronous data transfers and kernels to the CUDA stream
    // returned by cms::cuda::ScopedContextAcquire::stream()
    gpuAlgo_.makeAsync(raw, buffers, conditionsWrapper.get(), ctx.stream());

    // Destructor of ctx queues a callback to the CUDA stream notifying
    // waitingTaskHolder when the queued asynchronous work has finished
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    cms::cuda::ScopedContextProduce ctx{ctxState_};

    auto output = gpuAlgo_.getResults(ctx.stream());

    ev.put(std::move(output));

    for (auto& buf : buffers)
      buf.reset(nullptr);
  }

private:
  void initialize(const edm::EventSetup& es);
  void run(const FEDRawDataCollection& rawColl);
  void fill(uint32_t idet, const FEDRawDataCollection& rawColl);

private:
  std::vector<std::unique_ptr<sistrip::FEDBuffer>> buffers;
  std::vector<const FEDRawData*> raw;
  cms::cuda::ContextState ctxState_;

  stripgpu::SiStripRawToClusterGPUKernel gpuAlgo_;
  std::unique_ptr<SiStripConditionsGPUWrapper> conditionsWrapper;

  edm::EDGetTokenT<FEDRawDataCollection> productToken_;

  SiStripDetCabling const* cabling_;

  std::unique_ptr<StripClusterizerAlgorithm> clusterizer_;
  std::unique_ptr<SiStripRawProcessingAlgorithms> rawAlgos_;

  // March 2012: add flag for disabling APVe check in configuration
  bool doAPVEmulatorCheck_;

  bool legacy_;
  bool hybridZeroSuppressed_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClusterizerFromRawGPU);

void SiStripClusterizerFromRawGPU::initialize(const edm::EventSetup& es) {
  (*clusterizer_).initialize(es);
  cabling_ = (*clusterizer_).cabling();
  (*rawAlgos_).initialize(es);
}

void SiStripClusterizerFromRawGPU::run(const FEDRawDataCollection& rawColl) {
  // loop over good det in cabling
  for (auto idet : clusterizer_->allDetIds()) {
    fill(idet, rawColl);
  }  // end loop over dets
}

void SiStripClusterizerFromRawGPU::fill(uint32_t idet, const FEDRawDataCollection& rawColl) {

  auto const& det = clusterizer_->findDetId(idet);
  if (!det.valid())
    return;

  // Loop over apv-pairs of det
  for (auto const conn : clusterizer_->currentConnection(det)) {
    if
      UNLIKELY(!conn) continue;

    const uint16_t fedId = conn->fedId();

    // If fed id is null or connection is invalid continue
    if
      UNLIKELY(!fedId || !conn->isConnected()) { continue; }

    // If Fed hasnt already been initialised, extract data and initialise
    sistrip::FEDBuffer* buffer = buffers[fedId].get();
    if (!buffer) {
      const FEDRawData& rawData = rawColl.FEDData(fedId);
      raw[fedId] = &rawData;
      buffer = fillBuffer(fedId, rawData).release();
      if (!buffer) {
        continue;
      }
      buffers[fedId].reset(buffer);
    }
    assert(buffer);

    buffer->setLegacyMode(legacy_);
  }  // end loop over conn
}
