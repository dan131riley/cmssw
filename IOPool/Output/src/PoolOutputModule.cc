#include "IOPool/Output/interface/PoolOutputModule.h"

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

#include <fstream>
#include <iomanip>
#include <sstream>
#include "boost/algorithm/string.hpp"


namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
    edm::one::OutputModuleBase::OutputModuleBase(pset),
    one::OutputModule<WatchInputFiles>(pset),
    PoolOutputModuleBase(pset, wantAllEvents()),
    rootOutputFile_() {
  }

  void PoolOutputModule::beginJob() {
    beginJobBase();
  }

  bool PoolOutputModule::OMwantAllEvents() const { return wantAllEvents(); }
  BranchIDLists const* PoolOutputModule::OMbranchIDLists() { return branchIDLists(); }
  ThinnedAssociationsHelper const* PoolOutputModule::OMthinnedAssociationsHelper() const { return thinnedAssociationsHelper(); }
  ParameterSetID PoolOutputModule::OMselectorConfig() const { return selectorConfig(); }
  SelectedProductsForBranchType const& PoolOutputModule::OMkeptProducts() const { return keptProducts(); }

  std::string const& PoolOutputModule::currentFileName() const {
    return rootOutputFile_->fileName();
  }

  void PoolOutputModule::beginInputFile(FileBlock const& fb) {
    if(isFileOpen()) {
      beginInputFileBase(fb);
      rootOutputFile_->beginInputFile(fb, remainingEvents());
    }
  }

  void PoolOutputModule::openFile(FileBlock const& fb) {
    if(!isFileOpen()) {
      reallyOpenFile();
      beginInputFile(fb);
    }
  }

  void PoolOutputModule::respondToOpenInputFile(FileBlock const& fb) {
    auto init = initializedFromInput();
    respondToOpenInputFileBase(fb, keptProducts());
    beginInputFile(fb);
    if (isFileOpen() && !init) rootOutputFile_->fillSelectedProductList();
  }

  void PoolOutputModule::respondToCloseInputFile(FileBlock const& fb) {
    if(rootOutputFile_) rootOutputFile_->respondToCloseInputFile(fb);
  }

  PoolOutputModule::~PoolOutputModule() {
  }

  void PoolOutputModule::write(EventForOutput const& e) {
    updateBranchParents(e, subProcessParentageHelper());
    rootOutputFile_->writeOne(e);
      if (!statusFileName().empty()) {
        std::ofstream statusFile(statusFileName().c_str());
        statusFile << e.id() << " time: " << std::setprecision(3) << TimeOfDay() << '\n';
        statusFile.close();
      }
  }

  void PoolOutputModule::writeLuminosityBlock(LuminosityBlockForOutput const& lb) {
    rootOutputFile_->writeLuminosityBlock(lb);
  }

  void PoolOutputModule::writeRun(RunForOutput const& r) {
    rootOutputFile_->writeRun(r);
  }

  void PoolOutputModule::reallyCloseFile() {
    reallyCloseFileBase(rootOutputFile_);
    rootOutputFile_ = nullptr;
  }
  bool PoolOutputModule::isFileOpen() const { return rootOutputFile_.get() != nullptr; }
  bool PoolOutputModule::shouldWeCloseFile() const { return rootOutputFile_->shouldWeCloseFile(); }

  void PoolOutputModule::reallyOpenFile() {
    auto names = physicalAndLogicalNameForNewFile();
    rootOutputFile_ = std::make_unique<RootOutputFile>(this, names.first, names.second); // propagate_const<T> has no reset() function
  }

  void
  PoolOutputModule::preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {
    preActionBeforeRunEventAsyncBase(iTask, iModuleCallingContext, iPrincipal);
  }

  void
  PoolOutputModule::fillDescription(ParameterSetDescription& desc) {
    PoolOutputModuleBase::fillDescription(desc);
    OutputModule::fillDescription(desc);
  }

  void
  PoolOutputModule::fillDescriptions(ConfigurationDescriptions & descriptions) {
    ParameterSetDescription desc;
    PoolOutputModule::fillDescription(desc);
    descriptions.add("edmOutput", desc);
  }
}
