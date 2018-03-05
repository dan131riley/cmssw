#include "IOPool/Output/src/ParallelPoolOutputModule.h"

#include "IOPool/Output/src/RootOutputFile.h"

#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/SubProcessParentageHelper.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "TTree.h"
#include "TBranchElement.h"
#include "TObjArray.h"
#include "RVersion.h"
#include "Compression.h"

#include "ROOT/TBufferMerger.hxx"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <future>
#include "boost/algorithm/string.hpp"

namespace edm {
  ParallelPoolOutputModule::ParallelPoolOutputModule(ParameterSet const& pset) :
    edm::limited::OutputModuleBase::OutputModuleBase(pset),
    limited::OutputModule<WatchInputFiles>(pset),
    PoolOutputModuleBase(pset, wantAllEvents()),
    rootOutputFile_(),
    eventOutputFiles_(),
    eventAutoSaveSize_(pset.getUntrackedParameter<int>("eventAutoSaveSize")),
    moduleLabel_(pset.getParameter<std::string>("@module_label")),
    taskArena_(std::make_shared<tbb::task_arena>(tbb::task_arena::attach())) {
      mergeExec_ = [this](const std::function<void()> &f){
        std::promise<void> barrier;
        auto fwrap = [&]() { 
          auto set_value = [](decltype(barrier)* b) { b->set_value(); };
          std::unique_ptr<decltype(barrier), decltype(set_value)> release(&barrier, set_value);
          f();
        };
        taskArena_->enqueue(fwrap);
        barrier.get_future().wait();
      };
      queueSizeHistogram_.resize(pset.getUntrackedParameter<unsigned int>("concurrencyLimit"));
    }

  ParallelPoolOutputModule::~ParallelPoolOutputModule() {
    // NOTE: bad idea?
    if (!eventOutputFiles_.empty()) {
      LogWarning("ParallelPoolOutputModule::~ParallelPoolOutputModule") << "eventOutputFiles_ not empty";
      EventFileRec outputFileRec;
      while (eventOutputFiles_.try_pop(outputFileRec)) {
        outputFileRec.eventFile_->writeEvents(true);
        outputFileRec.eventFile_ = nullptr;
      }
    }
  }

  void ParallelPoolOutputModule::beginJob() {
    beginJobBase();
  }

  bool ParallelPoolOutputModule::OMwantAllEvents() const { return wantAllEvents(); }
  BranchIDLists const* ParallelPoolOutputModule::OMbranchIDLists() {
    // only called via reallyCloseFile()
    return branchIDLists();
  }
  ThinnedAssociationsHelper const* ParallelPoolOutputModule::OMthinnedAssociationsHelper() const { return thinnedAssociationsHelper(); }
  ParameterSetID ParallelPoolOutputModule::OMselectorConfig() const { return selectorConfig(); }
  SelectedProductsForBranchType const& ParallelPoolOutputModule::OMkeptProducts() const { return keptProducts(); }

  std::string const& ParallelPoolOutputModule::currentFileName() const {
    return rootOutputFile_->fileName();
  }

  void ParallelPoolOutputModule::beginInputFile(FileBlock const& fb) {
    if(isFileOpen()) {
      beginInputFileBase(fb);
      rootOutputFile_->beginInputFile(fb, remainingEvents());
    }
  }

  //NOTE: assumed serialized by framework
  void ParallelPoolOutputModule::openFile(FileBlock const& fb) {
    if(!isFileOpen()) {
      reallyOpenFile();
      beginInputFile(fb);
    }
  }

  //NOTE: assumed serialized by framework
  void ParallelPoolOutputModule::respondToOpenInputFile(FileBlock const& fb) {
    auto init = initializedFromInput();
    respondToOpenInputFileBase(fb, keptProducts());
    beginInputFile(fb);
    if (isFileOpen() && !init) rootOutputFile_->fillSelectedProductList();
  }

  //NOTE: assumed serialized by framework
  void ParallelPoolOutputModule::respondToCloseInputFile(FileBlock const& fb) {
    if (rootOutputFile_) rootOutputFile_->respondToCloseInputFile(fb);
  }

  void ParallelPoolOutputModule::write(EventForOutput const& e) {
    updateBranchParents(e, subProcessParentageHelper());

    // NOTE: Order matters here, sentry MUST be destroyed before outputFileRec
    EventFileRec outputFileRec;

    auto pushfile = [&](decltype(outputFileRec)* f) { eventOutputFiles_.push(std::move(*f)); };
    std::unique_ptr<decltype(outputFileRec), decltype(pushfile)> sentry(&outputFileRec, pushfile);

    if (!eventOutputFiles_.try_pop(outputFileRec)) {
      auto names = physicalAndLogicalNameForNewFile();
      outputFileRec.eventFile_ = std::make_unique<RootOutputFile>(this, names.first, names.second, mergePtr_->GetFile());
    }

    ++queueSizeHistogram_[eventOutputFiles_.size()];

    outputFileRec.eventFile_->writeOne(e);
    outputFileRec.eventFile_->writeEvents();

    outputFileRec.entries_ = outputFileRec.eventFile_->getEntries(edm::poolNames::eventTreeName());

    if (mergePtr_->GetQueueSize() > 1) {
      LogSystem(moduleLabel_) << "TBufferMerger Queue size " << mergePtr_->GetQueueSize();
    }

    if (!statusFileName().empty()) {
      std::lock_guard<std::mutex> lock{notYetThreadSafe_}; // NOTE: urrrggggghhhhh...
      std::ofstream statusFile(statusFileName().c_str());
      statusFile << e.id() << " time: " << std::setprecision(3) << TimeOfDay() << '\n';
      statusFile.close();
    }
  }

  void ParallelPoolOutputModule::writeLuminosityBlock(LuminosityBlockForOutput const& lb) {
    std::lock_guard<std::mutex> lock{notYetThreadSafe_};
    rootOutputFile_->writeLuminosityBlock(lb);
  }

  void ParallelPoolOutputModule::writeRun(RunForOutput const& r) {
    std::lock_guard<std::mutex> lock{notYetThreadSafe_};
    rootOutputFile_->writeRun(r);
  }

  //NOTE: assumed serialized by framework
  void ParallelPoolOutputModule::reallyCloseFile() {
    EventFileRec outputFileRec;

    LogSystem(moduleLabel_) << "Queue size " << eventOutputFiles_.size();
    for (size_t i = 0; i < queueSizeHistogram_.size(); ++i) {
      LogAbsolute(moduleLabel_) << std::setw(6) << i << " : " << std::setw(6) << queueSizeHistogram_[i];
    }
    //NOTE: need to merge the provenance from the writers before deleting!
    while (eventOutputFiles_.try_pop(outputFileRec)) {
      outputFileRec.eventFile_->writeEvents(true);
      outputFileRec.eventFile_ = nullptr;
    }
    reallyCloseFileBase(rootOutputFile_, true);
    rootOutputFile_ = nullptr;
    mergePtr_ = nullptr;
  }
  bool ParallelPoolOutputModule::isFileOpen() const { return rootOutputFile_.get() != nullptr; }
  bool ParallelPoolOutputModule::shouldWeCloseFile() const { return rootOutputFile_->shouldWeCloseFile(); }

  void ParallelPoolOutputModule::reallyOpenFile() {
    auto names = physicalAndLogicalNameForNewFile();
    ROOT::ECompressionAlgorithm alg;
    if (compressionAlgorithm() == std::string("ZLIB")) {
      alg = ROOT::kZLIB;
    } else if (compressionAlgorithm() == std::string("LZMA")) {
      alg = ROOT::kLZMA;
    } else {
      throw Exception(errors::Configuration) << "PoolOutputModuleBase configured with unknown compression algorithm '" << compressionAlgorithm() << "'\n"
					     << "Allowed compression algorithms are ZLIB and LZMA\n";
    }
    alg = ROOT::kZLIB; // TMP
    auto compress = ROOT::CompressionSettings(alg, compressionLevel());
    mergePtr_ = std::make_shared<ROOT::Experimental::TBufferMerger>(names.first.c_str(), "recreate", compress);
    mergePtr_->SetAutoSave(eventAutoSaveSize_);
    mergePtr_->RegisterCallback(mergeExec_);
    rootOutputFile_ = std::make_unique<RootOutputFile>(this, names.first, names.second, mergePtr_->GetFile());
  }

  //NOTE: assumed serialized by framework
  void
  ParallelPoolOutputModule::preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {
    preActionBeforeRunEventAsyncBase(iTask, iModuleCallingContext, iPrincipal);
  }

  void
  ParallelPoolOutputModule::fillDescription(ParameterSetDescription& desc) {
    desc.addUntracked<int>("eventAutoSaveSize",0)->setComment("Sets the ROOT TBufferMerger auto save size (in bytes) for the event TTree. The value sets how large the TBufferMerger queue must get before the queue is merged to the output file.  Large values reduce the overhead writing TTree AutoSave headers but also increase buffer memory use.");
    PoolOutputModuleBase::fillDescription(desc);
    OutputModule::fillDescription(desc);
  }

  void
  ParallelPoolOutputModule::fillDescriptions(ConfigurationDescriptions & descriptions) {
    ParameterSetDescription desc;
    ParallelPoolOutputModule::fillDescription(desc);
    descriptions.add("edmParallelOutput", desc);
  }
}
