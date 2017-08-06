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

// not in our ROOT version yet, so private copy
#include "IOPool/Output/src/TBufferMerger.hxx"

#include <fstream>
#include <iomanip>
#include <sstream>
#include "boost/algorithm/string.hpp"

namespace edm {
  ParallelPoolOutputModule::ParallelPoolOutputModule(ParameterSet const& pset) :
    edm::global::OutputModuleBase::OutputModuleBase(pset),
    global::OutputModule<WatchInputFiles>(pset),
    PoolOutputModuleBase(pset, wantAllEvents()),
    rootOutputFile_() {}

  void ParallelPoolOutputModule::beginJob() {
    beginJobBase();
  }

  bool ParallelPoolOutputModule::OMwantAllEvents() const { return wantAllEvents(); }
  BranchIDLists const* ParallelPoolOutputModule::OMbranchIDLists() { return branchIDLists(); }
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

  void ParallelPoolOutputModule::openFile(FileBlock const& fb) {
    if(!isFileOpen()) {
      reallyOpenFile();
      beginInputFile(fb);
    }
  }

  void ParallelPoolOutputModule::respondToOpenInputFile(FileBlock const& fb) {
    auto init = initializedFromInput();
    respondToOpenInputFileBase(fb, keptProducts());
    beginInputFile(fb);
    if (isFileOpen() && !init) rootOutputFile_->fillSelectedProductList();
  }

  void ParallelPoolOutputModule::respondToCloseInputFile(FileBlock const& fb) {
    if (rootOutputFile_) rootOutputFile_->respondToCloseInputFile(fb);
  }

  void ParallelPoolOutputModule::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    postForkReacquireResourcesBase(iChildIndex, iNumberOfChildren);
  }

  ParallelPoolOutputModule::~ParallelPoolOutputModule() {
  }

  void ParallelPoolOutputModule::write(EventForOutput const& e) {
    // ugly...should learn a prettier way
    struct FileRecHolder {
      FileRecHolder(EventOutputFiles& q) : queue_(q) {}
      ~FileRecHolder() {
        if (nullptr != fileRec_.eventFile_) queue_.push(std::move(fileRec_));
      }
      EventFileRec fileRec_;
      EventOutputFiles& queue_;
    };

    // NOTE: subProcessParentageHelper() not implemented in global::OutputModuleBase
    //updateBranchParents(e, subProcessParentageHelper());
    updateBranchParents(e, nullptr);

    FileRecHolder fileRecHolder(eventOutputFiles_);
    EventFileRec& fileRec = fileRecHolder.fileRec_;

    if (!eventOutputFiles_.try_pop(fileRec)) {
      auto names = physicalAndLogicalNameForNewFile();
      fileRec.eventFile_ = std::make_unique<RootOutputFile>(this, names.first, names.second, mergePtr_->GetFile());
      fileRec.fileCounter_ = eventFileCount_++;
    }

    auto rootfile = fileRec.eventFile_.get();
    rootfile->writeOne(e);
    rootfile->writeEvents();

    if (!statusFileName().empty()) {
      // NOTE: sigh...mutex here?
      std::ofstream statusFile(statusFileName().c_str());
      statusFile << e.id() << " time: " << std::setprecision(3) << TimeOfDay() << '\n';
      statusFile.close();
    }
  }

  void ParallelPoolOutputModule::writeLuminosityBlock(LuminosityBlockForOutput const& lb) {
    rootOutputFile_->writeLuminosityBlock(lb);
  }

  void ParallelPoolOutputModule::writeRun(RunForOutput const& r) {
    rootOutputFile_->writeRun(r);
  }

  void ParallelPoolOutputModule::reallyCloseFile() {
    EventFileRec frec;
    while (eventOutputFiles_.try_pop(frec)) {
      frec.eventFile_->writeEvents(true);
      frec.eventFile_ = nullptr;
    }

    reallyCloseFileBase(rootOutputFile_, true);
    rootOutputFile_ = nullptr;
    mergePtr_ = nullptr;
    LogWarning("ParallelPoolOutputModule") << "Event file count " << eventFileCount_;
    eventFileCount_ = 0;
  }
  bool ParallelPoolOutputModule::isFileOpen() const { return rootOutputFile_.get() != nullptr; }
  bool ParallelPoolOutputModule::shouldWeCloseFile() const { return rootOutputFile_->shouldWeCloseFile(); }

  void ParallelPoolOutputModule::reallyOpenFile() {
    auto names = physicalAndLogicalNameForNewFile();
    mergePtr_ = std::make_shared<ROOT::TBufferMerger>(names.first.c_str(), "recreate", compressionLevel());
    rootOutputFile_ = std::make_unique<RootOutputFile>(this, names.first, names.second, mergePtr_->GetFile());
  }


// NOTE: Not implemented in global::OutputModuleBase
/*
  void
  ParallelPoolOutputModule::preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {
    preActionBeforeRunEventAsyncBase(iTask, iModuleCallingContext, iPrincipal);
  }
*/

  void
  ParallelPoolOutputModule::fillDescription(ParameterSetDescription& desc) {
    PoolOutputModuleBase::fillDescription(desc);
    OutputModule::fillDescription(desc);
  }

  void
  ParallelPoolOutputModule::fillDescriptions(ConfigurationDescriptions & descriptions) {
    ParameterSetDescription desc;
    ParallelPoolOutputModule::fillDescription(desc);
    descriptions.add("edmOutput", desc);
  }
}
