#include "L1Trigger/TrackerTFP/interface/HoughTransform.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>
#include <cmath>

namespace trackerTFP {

  HoughTransform::HoughTransform(const tt::Setup* setup,
                                 const DataFormats* dataFormats,
                                 const LayerEncoding* layerEncoding,
                                 std::vector<StubHT>& stubs)
      : setup_(setup),
        dataFormats_(dataFormats),
        layerEncoding_(layerEncoding),
        inv2R_(&dataFormats_->format(Variable::inv2R, Process::ht)),
        phiT_(&dataFormats_->format(Variable::phiT, Process::ht)),
        zT_(&dataFormats_->format(Variable::zT, Process::gp)),
        phi_(&dataFormats_->format(Variable::phi, Process::ht)),
        z_(&dataFormats_->format(Variable::z, Process::gp)),
        stubs_(stubs) {
    numChannelIn_ = dataFormats_->numChannel(Process::gp);
    numChannelOut_ = dataFormats_->numChannel(Process::ht);
    chan_ = setup_->kfNumWorker();
    mux_ = numChannelOut_ / chan_;
  }

  // fill output products
  void HoughTransform::produce(const std::vector<std::vector<StubGP*>>& streamsIn,
                               std::vector<std::deque<StubHT*>>& streamsOut) {
    // count and reserve ht stubs
    auto multiplicity = [](int sum, StubGP* s) { return sum + (s ? 1 + s->inv2RMax() - s->inv2RMin() : 0); };
    int nStubs(0);
    for (const std::vector<StubGP*>& input : streamsIn)
      nStubs += std::accumulate(input.begin(), input.end(), 0, multiplicity);
    stubs_.reserve(nStubs);
    for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
      const int inv2Ru = mux_ * (channelOut % chan_) + channelOut / chan_;
      const int inv2R = inv2R_->toSigned(inv2Ru);
      std::deque<StubHT*>& output = streamsOut[channelOut];
      for (int channelIn = numChannelIn_ - 1; channelIn >= 0; channelIn--) {
        const std::vector<StubGP*>& input = streamsIn[channelIn];
        std::vector<StubHT*> stubs;
        stubs.reserve(2 * input.size());
        // associate stubs with inv2R and phiT bins
        fillIn(inv2R, channelIn, input, stubs);
        // apply truncation
        if (setup_->enableTruncation() && static_cast<int>(stubs.size()) > setup_->numFramesHigh())
          stubs.resize(setup_->numFramesHigh());
        // ht collects all stubs before readout starts -> remove all gaps
        stubs.erase(std::remove(stubs.begin(), stubs.end(), nullptr), stubs.end());
        // identify tracks
        readOut(stubs, output);
      }
      // apply truncation
      if (setup_->enableTruncation() && static_cast<int>(output.size()) > setup_->numFramesHigh())
        output.resize(setup_->numFramesHigh());
      // remove trailing gaps
      for (auto it = output.end(); it != output.begin();)
        it = (*--it) ? output.begin() : output.erase(it);
    }
  }

  // associate stubs with phiT bins in this inv2R column
  void HoughTransform::fillIn(int inv2R, int sector, const std::vector<StubGP*>& input, std::vector<StubHT*>& output) {
    const DataFormat& gp = dataFormats_->format(Variable::phiT, Process::gp);
    auto inv2RrangeCheck = [inv2R](StubGP* stub) {
      return (stub && stub->inv2RMin() <= inv2R && stub->inv2RMax() >= inv2R) ? stub : nullptr;
    };
    const int gpPhiT = gp.toSigned(sector % setup_->gpNumBinsPhiT());
    const int zT = sector / setup_->gpNumBinsPhiT() - setup_->gpNumBinsZT() / 2;
    const double inv2Rf = inv2R_->floating(inv2R);
    auto convert = [this, inv2Rf, gpPhiT, zT, gp](StubGP* stub, int phiTht, double phi, double z) {
      const double phiTf = phiT_->floating(phiTht);
      const int phiT = phiT_->integer(gp.floating(gpPhiT) + phiTf);
      const double htPhi = phi - (inv2Rf * stub->r() + phiTf);
      stubs_.emplace_back(*stub, stub->r(), htPhi, z, stub->layer(), phiT, zT);
      return &stubs_.back();
    };
    // Latency of ht fifo firmware
    static constexpr int latency = 1;
    // static delay container
    std::deque<StubHT*> delay(latency, nullptr);
    // fifo, used to store stubs which belongs to a second possible track
    std::deque<StubHT*> stack;
    // stream of incroming stubs
    std::deque<StubGP*> stream;
    std::transform(input.begin(), input.end(), std::back_inserter(stream), inv2RrangeCheck);
    // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
    while (!stream.empty() || !stack.empty() ||
           !std::all_of(delay.begin(), delay.end(), [](const StubHT* stub) { return !stub; })) {
      StubHT* stubHT = nullptr;
      StubGP* stubGP = pop_front(stream);
      if (stubGP) {
        const double phi = stubGP->phi();
        const double z = stubGP->z();
        const double phiT = phi - inv2Rf * stubGP->r();
        const int major = phiT_->integer(phiT);
        if (major >= -setup_->htNumBinsPhiT() / 2 && major < setup_->htNumBinsPhiT() / 2) {
          // major candidate has pt > threshold (3 GeV)
          stubHT = convert(stubGP, major, phi, z);
        }
        const double chi = phi_->digi(phiT - phiT_->floating(major));
        if (abs(stubGP->r() * inv2R_->base()) + 2. * abs(chi) >= phiT_->base()) {
          // stub belongs to two candidates
          const int minor = chi >= 0. ? major + 1 : major - 1;
          if (minor >= -setup_->htNumBinsPhiT() / 2 && minor < setup_->htNumBinsPhiT() / 2) {
            // second (minor) candidate has pt > threshold (3 GeV)
            StubHT* stub = convert(stubGP, minor, phi, z);
            delay.push_back(stub);
          }
        }
      }
      // add nullptr to delay pipe if stub didn't fill any cell
      if (static_cast<int>(delay.size()) == latency)
        delay.push_back(nullptr);
      // take fifo latency into account (read before write)
      StubHT* stub = pop_front(delay);
      if (stub) {
        // buffer overflow
        if (setup_->enableTruncation() && static_cast<int>(stack.size()) == setup_->htDepthMemory() - 1)
          pop_front(stack);
        // store minor stub in fifo
        stack.push_back(stub);
      }
      // take a minor stub if no major stub available
      output.push_back(stubHT ? stubHT : pop_front(stack));
    }
  }

  // identify tracks
  void HoughTransform::readOut(const std::vector<StubHT*>& input, std::deque<StubHT*>& output) const {
    auto toBinPhiT = [this](StubHT* stub) {
      const DataFormat& gp = dataFormats_->format(Variable::phiT, Process::gp);
      const double phiT = phiT_->floating(stub->phiT());
      const double local = phiT - gp.digi(phiT);
      return phiT_->integer(local) + setup_->htNumBinsPhiT() / 2;
    };
    auto toLayerId = [this](StubHT* stub) {
      const DataFormat& layer = dataFormats_->format(Variable::layer, Process::ctb);
      return stub->layer().val(layer.width());
    };
    // used to recognise in which order tracks are found
    TTBV trkFoundPhiTs(0, setup_->htNumBinsPhiT());
    // hitPattern for all possible tracks, used to find tracks
    std::vector<TTBV> patternHits(setup_->htNumBinsPhiT(), TTBV(0, setup_->numLayers()));
    // found phiTs, ordered in time
    std::vector<int> phiTs;
    phiTs.reserve(setup_->htNumBinsPhiT());
    for (StubHT* stub : input) {
      const int binPhiT = toBinPhiT(stub);
      const int layerId = toLayerId(stub);
      TTBV& pattern = patternHits[binPhiT];
      pattern.set(layerId);
      if (trkFoundPhiTs[binPhiT] || noTrack(pattern, stub->zT()))
        continue;
      // first time track found
      trkFoundPhiTs.set(binPhiT);
      phiTs.push_back(binPhiT);
    }
    // read out found tracks ordered as found
    for (int phiT : phiTs) {
      auto samePhiT = [phiT, toBinPhiT](StubHT* stub) { return toBinPhiT(stub) == phiT; };
      // read out stubs in reverse order to emulate f/w (backtracking linked list)
      std::copy_if(input.rbegin(), input.rend(), std::back_inserter(output), samePhiT);
    }
  }

  // track identification
  bool HoughTransform::noTrack(const TTBV& pattern, int zT) const {
    // not enough seeding layer
    if (pattern.count(0, setup_->kfMaxSeedingLayer()) < 2)
      return true;
    // check min layers req
    const int minLayers =
        setup_->htMinLayers() - (((zT == -4 || zT == 3) && (!pattern.test(5) && !pattern.test(7))) ? 1 : 0);
    // prepare pattern analysis
    const TTBV& maybePattern = layerEncoding_->maybePattern(zT);
    int nHits(0);
    int nGaps(0);
    bool doubleGap = false;
    for (int layer = 0; layer < setup_->numLayers(); layer++) {
      if (pattern.test(layer)) {
        doubleGap = false;
        if (++nHits == minLayers)
          return false;
      } else if (nHits < setup_->kfMinLayers() && !maybePattern.test(layer)) {
        if (++nGaps == setup_->kfMaxGaps() || doubleGap)
          break;
        doubleGap = true;
      }
    }
    return true;
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* HoughTransform::pop_front(std::deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trackerTFP
