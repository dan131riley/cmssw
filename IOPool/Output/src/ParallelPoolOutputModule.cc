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
#include "boost/algorithm/string.hpp"

namespace edm {
  ParallelPoolOutputModule::ParallelPoolOutputModule(ParameterSet const& pset) :
    edm::limited::OutputModuleBase::OutputModuleBase(pset),
    limited::OutputModule<WatchInputFiles>(pset),
    PoolOutputModuleBase(pset, wantAllEvents()),
    rootOutputFile_(),
    eventOutputFiles_(),
    moduleLabel_(pset.getParameter<std::string>("@module_label")) {}

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
      outputFileRec.fileIndex_ = eventFileCount_++;
    }
    outputFileRec.eventFile_->writeOne(e);
    outputFileRec.eventFile_->writeEvents();

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

    LogSystem(moduleLabel_) << "Event file count " << eventFileCount_ << " queue size " << eventOutputFiles_.size();
    //NOTE: need to merge the provenance from the writers before deleting!
    while (eventOutputFiles_.try_pop(outputFileRec)) {
      outputFileRec.eventFile_->writeEvents(true);
      outputFileRec.eventFile_ = nullptr;
    }
    reallyCloseFileBase(rootOutputFile_, true);
    rootOutputFile_ = nullptr;
    mergePtr_ = nullptr;
    eventFileCount_ = 0;
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
    mergePtr_ = std::make_shared<ROOT::Experimental::TBufferMerger>(names.first.c_str(), "recreate",
                                                      ROOT::CompressionSettings(alg, compressionLevel()));
    rootOutputFile_ = std::make_unique<RootOutputFile>(this, names.first, names.second, mergePtr_->GetFile());
  }

  //NOTE: assumed serialized by framework
  void
  ParallelPoolOutputModule::preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {
    preActionBeforeRunEventAsyncBase(iTask, iModuleCallingContext, iPrincipal);
  }

  void
  ParallelPoolOutputModule::fillDescription(ParameterSetDescription& desc) {
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
