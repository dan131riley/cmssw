#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/TrackerTFP/interface/Demonstrator.h"

#include <sstream>

namespace trackerTFP {

  /*! \class  trackerTFP::AnalyzerDemonstrator
   *  \brief  calls questasim to simulate the f/w and compares the results with clock-and-bit-accurate emulation.
   *          A single bit error interrupts the run.
   *  \author Thomas Schuh
   *  \date   2020, Nov
   */
  class AnalyzerDemonstrator : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
  public:
    AnalyzerDemonstrator(const edm::ParameterSet& iConfig);
    void beginJob() override {}
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override {}

  private:
    //
    void convert(const edm::Event& iEvent,
                 const edm::EDGetTokenT<tt::StreamsTrack>& tokenTracks,
                 const edm::EDGetTokenT<tt::StreamsStub>& tokenStubs,
                 std::vector<std::vector<tt::Frame>>& bits) const;
    //
    template <typename T>
    void convert(const T& collection, std::vector<std::vector<tt::Frame>>& bits) const;
    // ED input token of Tracks
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubsIn_;
    edm::EDGetTokenT<tt::StreamsStub> edGetTokenStubsOut_;
    // ED input token of Stubs
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracksIn_;
    edm::EDGetTokenT<tt::StreamsTrack> edGetTokenTracksOut_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;
    // Demonstrator token
    edm::ESGetToken<Demonstrator, tt::SetupRcd> esGetTokenDemonstrator_;
    //
    const tt::Setup* setup_ = nullptr;
    //
    const Demonstrator* demonstrator_ = nullptr;
  };

  AnalyzerDemonstrator::AnalyzerDemonstrator(const edm::ParameterSet& iConfig) {
    // book in- and output ED products
    const std::string& labelIn = iConfig.getParameter<std::string>("LabelIn");
    const std::string& labelOut = iConfig.getParameter<std::string>("LabelOut");
    const std::string& branchStubs = iConfig.getParameter<std::string>("BranchStubs");
    const std::string& branchTracks = iConfig.getParameter<std::string>("BranchTracks");
    edGetTokenStubsIn_ = consumes<tt::StreamsStub>(edm::InputTag(labelIn, branchStubs));
    if (labelOut != "ProducerTFP")
      edGetTokenStubsOut_ = consumes<tt::StreamsStub>(edm::InputTag(labelOut, branchStubs));
    if (labelIn == "ProducerCTB" || labelIn == "ProducerKF" || labelIn == "ProducerDR")
      edGetTokenTracksIn_ = consumes<tt::StreamsTrack>(edm::InputTag(labelIn, branchTracks));
    if (labelOut == "ProducerCTB" || labelOut == "ProducerKF" || labelOut == "ProducerDR" || labelOut == "ProducerTFP")
      edGetTokenTracksOut_ = consumes<tt::StreamsTrack>(edm::InputTag(labelOut, branchTracks));
    // book ES products
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
    esGetTokenDemonstrator_ = esConsumes<edm::Transition::BeginRun>();
  }

  void AnalyzerDemonstrator::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    //
    setup_ = &iSetup.getData(esGetTokenSetup_);
    //
    demonstrator_ = &iSetup.getData(esGetTokenDemonstrator_);
  }

  void AnalyzerDemonstrator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    edm::Handle<tt::StreamsStub> handle;
    iEvent.getByToken<tt::StreamsStub>(edGetTokenStubsIn_, handle);
    std::vector<std::vector<tt::Frame>> input;
    std::vector<std::vector<tt::Frame>> output;
    convert(iEvent, edGetTokenTracksIn_, edGetTokenStubsIn_, input);
    convert(iEvent, edGetTokenTracksOut_, edGetTokenStubsOut_, output);
    if (!demonstrator_->analyze(input, output)) {
      cms::Exception exception("RunTimeError.");
      exception.addContext("trackerTFP::AnalyzerDemonstrator::analyze");
      exception << "Bit error detected.";
      throw exception;
    }
  }

  //
  void AnalyzerDemonstrator::convert(const edm::Event& iEvent,
                                     const edm::EDGetTokenT<tt::StreamsTrack>& tokenTracks,
                                     const edm::EDGetTokenT<tt::StreamsStub>& tokenStubs,
                                     std::vector<std::vector<tt::Frame>>& bits) const {
    const bool tracks = !tokenTracks.isUninitialized();
    const bool stubs = !tokenStubs.isUninitialized();
    edm::Handle<tt::StreamsStub> handleStubs;
    edm::Handle<tt::StreamsTrack> handleTracks;
    int numChannelStubs(0);
    if (stubs) {
      iEvent.getByToken<tt::StreamsStub>(tokenStubs, handleStubs);
      numChannelStubs = handleStubs->size();
    }
    int numChannelTracks(0);
    if (tracks) {
      iEvent.getByToken<tt::StreamsTrack>(tokenTracks, handleTracks);
      numChannelTracks = handleTracks->size();
    }
    numChannelTracks /= setup_->numRegions();
    numChannelStubs /= (setup_->numRegions() * (tracks ? numChannelTracks : 1));
    bits.reserve(numChannelTracks + numChannelStubs);
    for (int region = 0; region < setup_->numRegions(); region++) {
      if (tracks) {
        const int offsetTracks = region * numChannelTracks;
        for (int channelTracks = 0; channelTracks < numChannelTracks; channelTracks++) {
          const int offsetStubs = (region * numChannelTracks + channelTracks) * numChannelStubs;
          if (tracks)
            convert(handleTracks->at(offsetTracks + channelTracks), bits);
          if (stubs) {
            for (int channelStubs = 0; channelStubs < numChannelStubs; channelStubs++)
              convert(handleStubs->at(offsetStubs + channelStubs), bits);
          }
        }
      } else {
        const int offsetStubs = region * numChannelStubs;
        for (int channelStubs = 0; channelStubs < numChannelStubs; channelStubs++)
          convert(handleStubs->at(offsetStubs + channelStubs), bits);
      }
    }
  }

  //
  template <typename T>
  void AnalyzerDemonstrator::convert(const T& collection, std::vector<std::vector<tt::Frame>>& bits) const {
    bits.emplace_back();
    std::vector<tt::Frame>& bvs = bits.back();
    bvs.reserve(collection.size());
    std::transform(
        collection.begin(), collection.end(), std::back_inserter(bvs), [](const auto& frame) { return frame.second; });
  }

}  // namespace trackerTFP

DEFINE_FWK_MODULE(trackerTFP::AnalyzerDemonstrator);
