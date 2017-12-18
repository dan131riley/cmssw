#include "IOPool/Output/interface/PoolOutputModuleBase.h"

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
#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "TTree.h"
#include "TBranchElement.h"
#include "TObjArray.h"
#include "RVersion.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include "boost/algorithm/string.hpp"

namespace edm {
  PoolOutputModuleBase::PoolOutputModuleBase(ParameterSet const& pset, bool wantAllEvents) :
    rootServiceChecker_(),
    auxItems_(),
    selectedOutputItemList_(),
    fileName_(pset.getUntrackedParameter<std::string>("fileName")),
    logicalFileName_(pset.getUntrackedParameter<std::string>("logicalFileName")),
    catalog_(pset.getUntrackedParameter<std::string>("catalog")),
    maxFileSize_(pset.getUntrackedParameter<int>("maxSize")),
    compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel")),
    compressionAlgorithm_(pset.getUntrackedParameter<std::string>("compressionAlgorithm")),
    basketSize_(pset.getUntrackedParameter<int>("basketSize")),
    eventAutoFlushSize_(pset.getUntrackedParameter<int>("eventAutoFlushCompressedSize")),
    splitLevel_(std::min<int>(pset.getUntrackedParameter<int>("splitLevel") + 1, 99)),
    basketOrder_(pset.getUntrackedParameter<std::string>("sortBaskets")),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize")),
    whyNotFastClonable_(pset.getUntrackedParameter<bool>("fastCloning") ? FileBlock::CanFastClone : FileBlock::DisabledInConfigFile),
    dropMetaData_(DropNone),
    moduleLabel_(pset.getParameter<std::string>("@module_label")),
    initializedFromInput_(false),
    outputFileCount_(0),
    inputFileCount_(0),
    childIndex_(0U),
    numberOfDigitsInIndex_(0U),
    branchParents_(),
    branchChildren_(),
    overrideInputFileSplitLevels_(pset.getUntrackedParameter<bool>("overrideInputFileSplitLevels")),
    statusFileName_() {

      if (pset.getUntrackedParameter<bool>("writeStatusFile")) {
        std::ostringstream statusfilename;
        statusfilename << moduleLabel_ << '_' << getpid();
        statusFileName_ = statusfilename.str();
      }

      std::string dropMetaData(pset.getUntrackedParameter<std::string>("dropMetaData"));
      if(dropMetaData.empty()) dropMetaData_ = DropNone;
      else if(dropMetaData == std::string("NONE")) dropMetaData_ = DropNone;
      else if(dropMetaData == std::string("DROPPED")) dropMetaData_ = DropDroppedPrior;
      else if(dropMetaData == std::string("PRIOR")) dropMetaData_ = DropPrior;
      else if(dropMetaData == std::string("ALL")) dropMetaData_ = DropAll;
      else {
        throw edm::Exception(errors::Configuration, "Illegal dropMetaData parameter value: ")
            << dropMetaData << ".\n"
            << "Legal values are 'NONE', 'DROPPED', 'PRIOR', and 'ALL'.\n";
      }

    if (!wantAllEvents) {
      whyNotFastClonable_+= FileBlock::EventSelectionUsed;
    }

    auto const& specialSplit {pset.getUntrackedParameterSetVector("overrideBranchesSplitLevel")};
      
    specialSplitLevelForBranches_.reserve(specialSplit.size());
    for(auto const& s: specialSplit) {
      specialSplitLevelForBranches_.emplace_back(s.getUntrackedParameter<std::string>("branch"),
                                                 s.getUntrackedParameter<int>("splitLevel"));
    }
      
    // We don't use this next parameter, but we read it anyway because it is part
    // of the configuration of this module.  An external parser creates the
    // configuration by reading this source code.
    pset.getUntrackedParameterSet("dataset");
  }

  void PoolOutputModuleBase::beginJobBase() {
    Service<ConstProductRegistry> reg;
    for(auto const& prod : reg->productList()) {
      BranchDescription const& desc = prod.second;
      if (desc.produced() && desc.branchType() == InEvent && !desc.isAlias()) {
        producedBranches_.emplace_back(desc.branchID());
      }
    }
  }

  PoolOutputModuleBase::AuxItem::AuxItem() :
        basketSize_(BranchDescription::invalidBasketSize) {}

  PoolOutputModuleBase::OutputItem::OutputItem() :
        branchDescription_(nullptr),
        token_(),
        splitLevel_(BranchDescription::invalidSplitLevel),
        basketSize_(BranchDescription::invalidBasketSize) {}

  PoolOutputModuleBase::OutputItem::OutputItem(BranchDescription const* bd, EDGetToken const& token, int splitLevel, int basketSize) :
        branchDescription_(bd),
        token_(token),
        splitLevel_(splitLevel),
        basketSize_(basketSize) {}


  PoolOutputModuleBase::OutputItem::Sorter::Sorter(TTree* tree) : treeMap_(new std::map<std::string, int>) {
    // Fill a map mapping branch names to an index specifying the order in the tree.
    if(tree != nullptr) {
      TObjArray* branches = tree->GetListOfBranches();
      for(int i = 0; i < branches->GetEntries(); ++i) {
        TBranchElement* br = (TBranchElement*)branches->At(i);
        treeMap_->insert(std::make_pair(std::string(br->GetName()), i));
      }
    }
  }

  bool
  PoolOutputModuleBase::OutputItem::Sorter::operator()(OutputItem const& lh, OutputItem const& rh) const {
    // Provides a comparison for sorting branches according to the index values in treeMap_.
    // Branches not found are always put at the end (i.e. not found > found).
    if(treeMap_->empty()) return lh < rh;
    std::string const& lstring = lh.branchDescription_->branchName();
    std::string const& rstring = rh.branchDescription_->branchName();
    std::map<std::string, int>::const_iterator lit = treeMap_->find(lstring);
    std::map<std::string, int>::const_iterator rit = treeMap_->find(rstring);
    bool lfound = (lit != treeMap_->end());
    bool rfound = (rit != treeMap_->end());
    if(lfound && rfound) {
      return lit->second < rit->second;
    } else if(lfound) {
      return true;
    } else if(rfound) {
      return false;
    }
    return lh < rh;
  }
  
  inline bool PoolOutputModuleBase::SpecialSplitLevelForBranch::match( std::string const& iBranchName) const {
    return std::regex_match(iBranchName,branch_);
  }

  std::regex PoolOutputModuleBase::SpecialSplitLevelForBranch::convert( std::string const& iGlobBranchExpression) const {
    std::string tmp(iGlobBranchExpression);
    boost::replace_all(tmp, "*", ".*");
    boost::replace_all(tmp, "?", ".");
    return std::regex(tmp);
  }
  
  void PoolOutputModuleBase::fillSelectedItemList(BranchType branchType, TTree* theInputTree, SelectedProductsForBranchType const& keptProducts) {

    SelectedProducts const& keptVector = keptProducts[branchType];
    OutputItemList&   outputItemList = selectedOutputItemList_[branchType];
    AuxItem&   auxItem = auxItems_[branchType];

    // Fill AuxItem
    if (theInputTree != nullptr && !overrideInputFileSplitLevels_) {
      TBranch* auxBranch = theInputTree->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str());
      if (auxBranch) {
        auxItem.basketSize_ = auxBranch->GetBasketSize();
      } else {
        auxItem.basketSize_ = basketSize_;
      }
    } else {
      auxItem.basketSize_ = basketSize_;
    }

    // Fill outputItemList with an entry for each branch.
    for(auto const& kept : keptVector) {
      int splitLevel = BranchDescription::invalidSplitLevel;
      int basketSize = BranchDescription::invalidBasketSize;

      BranchDescription const& prod = *kept.first;
      TBranch* theBranch = ((!prod.produced() && theInputTree != nullptr && !overrideInputFileSplitLevels_) ? theInputTree->GetBranch(prod.branchName().c_str()) : nullptr);

      if(theBranch != nullptr) {
        splitLevel = theBranch->GetSplitLevel();
        basketSize = theBranch->GetBasketSize();
      } else {
        splitLevel = (prod.splitLevel() == BranchDescription::invalidSplitLevel ? splitLevel_ : prod.splitLevel());
        for(auto const& b: specialSplitLevelForBranches_) {
          if(b.match(prod.branchName())) {
            splitLevel =b.splitLevel_;
          }
        }
        basketSize = (prod.basketSize() == BranchDescription::invalidBasketSize ? basketSize_ : prod.basketSize());
      }
      outputItemList.emplace_back(&prod, kept.second, splitLevel, basketSize);
    }

    // Sort outputItemList to allow fast copying.
    // The branches in outputItemList must be in the same order as in the input tree, with all new branches at the end.
    sort_all(outputItemList, OutputItem::Sorter(theInputTree));
  }

  void PoolOutputModuleBase::beginInputFileBase(FileBlock const& fb) {
    //Faster to read ChildrenBranches directly from input
    // file than to build it every event
    auto const& branchToChildMap = fb.branchChildren().childLookup();
    for (auto const& parentToChildren : branchToChildMap) {
      for (auto const& child : parentToChildren.second) {
        branchChildren_.insertChild(parentToChildren.first, child);
      }
    }
  }

  void PoolOutputModuleBase::respondToOpenInputFileBase(FileBlock const& fb, SelectedProductsForBranchType const& keptProducts) {
    if(!initializedFromInput_) {
      for(int i = InEvent; i < NumBranchTypes; ++i) {
        BranchType branchType = static_cast<BranchType>(i);
        TTree* theInputTree = (branchType == InEvent ? fb.tree() :
                              (branchType == InLumi ? fb.lumiTree() :
                               fb.runTree()));
        fillSelectedItemList(branchType, theInputTree, keptProducts);
      }
      initializedFromInput_ = true;
    }
    ++inputFileCount_;
  }

  PoolOutputModuleBase::~PoolOutputModuleBase() noexcept (false) {
  }

  void PoolOutputModuleBase::reallyCloseFileBase(RootOutputFile* rootOutputFile, bool doWrite) {
    fillDependencyGraph();
    branchParents_.clear();
    startEndFile(rootOutputFile);
    writeFileFormatVersion(rootOutputFile);
    writeFileIdentifier(rootOutputFile);
    writeIndexIntoFile(rootOutputFile);
    writeProcessHistoryRegistry(rootOutputFile);
    writeParameterSetRegistry(rootOutputFile);
    writeProductDescriptionRegistry(rootOutputFile);
    writeParentageRegistry(rootOutputFile);
    writeBranchIDListRegistry(rootOutputFile);
    writeThinnedAssociationsHelper(rootOutputFile);
    writeProductDependencies(rootOutputFile); //branchChildren used here
    branchChildren_.clear();
    finishEndFile(rootOutputFile, doWrite);

    doExtrasAfterCloseFile();
  }

  // At some later date, we may move functionality from finishEndFile() to here.
  void PoolOutputModuleBase::startEndFile(RootOutputFile* rootOutputFile) { }

  void PoolOutputModuleBase::writeFileFormatVersion(RootOutputFile* rootOutputFile) { rootOutputFile->writeFileFormatVersion(); }
  void PoolOutputModuleBase::writeFileIdentifier(RootOutputFile* rootOutputFile) { rootOutputFile->writeFileIdentifier(); }
  void PoolOutputModuleBase::writeIndexIntoFile(RootOutputFile* rootOutputFile) { rootOutputFile->writeIndexIntoFile(); }
  void PoolOutputModuleBase::writeProcessHistoryRegistry(RootOutputFile* rootOutputFile) { rootOutputFile->writeProcessHistoryRegistry(); }
  void PoolOutputModuleBase::writeParameterSetRegistry(RootOutputFile* rootOutputFile) { rootOutputFile->writeParameterSetRegistry(); }
  void PoolOutputModuleBase::writeProductDescriptionRegistry(RootOutputFile* rootOutputFile) { rootOutputFile->writeProductDescriptionRegistry(); }
  void PoolOutputModuleBase::writeParentageRegistry(RootOutputFile* rootOutputFile) { rootOutputFile->writeParentageRegistry(); }
  void PoolOutputModuleBase::writeBranchIDListRegistry(RootOutputFile* rootOutputFile) { rootOutputFile->writeBranchIDListRegistry(); }
  void PoolOutputModuleBase::writeThinnedAssociationsHelper(RootOutputFile* rootOutputFile) { rootOutputFile->writeThinnedAssociationsHelper(); }
  void PoolOutputModuleBase::writeProductDependencies(RootOutputFile* rootOutputFile) { rootOutputFile->writeProductDependencies(); }
  void PoolOutputModuleBase::finishEndFile(RootOutputFile* rootOutputFile, bool doWrite) { rootOutputFile->finishEndFile(doWrite); }
  void PoolOutputModuleBase::doExtrasAfterCloseFile() {}

  std::pair<std::string, std::string>
  PoolOutputModuleBase::physicalAndLogicalNameForNewFile() {
      if(inputFileCount_ == 0) {
        throw edm::Exception(errors::LogicError)
          << "Attempt to open output file before input file. "
          << "Please report this to the core framework developers.\n";
      }
      std::string suffix(".root");
      std::string::size_type offset = fileName().rfind(suffix);
      bool ext = (offset == fileName().size() - suffix.size());
      if(!ext) suffix.clear();
      std::string fileBase(ext ? fileName().substr(0, offset) : fileName());
      std::ostringstream ofilename;
      std::ostringstream lfilename;
      ofilename << fileBase;
      lfilename << logicalFileName();
      if(numberOfDigitsInIndex_) {
        ofilename << '_' << std::setw(numberOfDigitsInIndex_) << std::setfill('0') << childIndex_;
        if(!logicalFileName().empty()) {
          lfilename << '_' << std::setw(numberOfDigitsInIndex_) << std::setfill('0') << childIndex_;
        }
      }
      if(outputFileCount_) {
        ofilename << std::setw(3) << std::setfill('0') << outputFileCount_;
        if(!logicalFileName().empty()) {
          lfilename << std::setw(3) << std::setfill('0') << outputFileCount_;
        }
      }
      ofilename << suffix;
      ++outputFileCount_;

      return std::make_pair(ofilename.str(), lfilename.str());
  }

  void
  PoolOutputModuleBase::updateBranchParentsForOneBranch(
    ProductProvenanceRetriever const* provRetriever,
    BranchID const& branchID) {

    ProductProvenance const* provenance = provRetriever->branchIDToProvenanceForProducedOnly(branchID);
    if (provenance != nullptr) {
      BranchParents::iterator it = branchParents_.find(branchID);
      if (it == branchParents_.end()) {
        it = branchParents_.insert(std::make_pair(branchID, ParentSet())).first;
      }
      it->second.insert(provenance->parentageID());
    }
  }

  void
  PoolOutputModuleBase::updateBranchParents(EventForOutput const& e, SubProcessParentageHelper const* helper) {

    ProductProvenanceRetriever const* provRetriever = e.productProvenanceRetrieverPtr();
    for (auto const& bid : producedBranches_) {
      updateBranchParentsForOneBranch(provRetriever, bid);
    }
    if (helper) {
      for (auto const& bid : helper->producedProducts()) {
        updateBranchParentsForOneBranch(provRetriever, bid);
      }
    }
  }

  void
  PoolOutputModuleBase::preActionBeforeRunEventAsyncBase(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {
    if(DropAll != dropMetaData_ ) {
      auto const* ep = dynamic_cast<EventPrincipal const*>(&iPrincipal);
      if(ep)
      {
        auto pr = ep->productProvenanceRetrieverPtr();
        if(pr) {
          pr->readProvenanceAsync(iTask,&iModuleCallingContext);
        }
      }
    }
  }

  void
  PoolOutputModuleBase::fillDependencyGraph() {
    for(auto const& branchParent : branchParents_) {
      BranchID const& child = branchParent.first;
      auto const& eIds = branchParent.second;
      for(auto const& eId : eIds) {
        Parentage entryDesc;
        ParentageRegistry::instance()->getMapped(eId, entryDesc);
        std::vector<BranchID> const& parents = entryDesc.parents();
        for(auto const& parent : parents) {
          branchChildren_.insertChild(parent, child);
        }
      }
    }
  }

  void
  PoolOutputModuleBase::fillDescription(ParameterSetDescription& desc) {
    std::string defaultString;

    desc.setComment("Writes runs, lumis, and events into EDM/ROOT files.");
    desc.addUntracked<std::string>("fileName")
        ->setComment("Name of output file.");
    desc.addUntracked<std::string>("logicalFileName", defaultString)
        ->setComment("Passed to job report. Otherwise unused by module.");
    desc.addUntracked<std::string>("catalog", defaultString)
        ->setComment("Passed to job report. Otherwise unused by module.");
    desc.addUntracked<int>("maxSize", 0x7f000000)
        ->setComment("Maximum output file size, in kB.\n"
                     "If over maximum, new output file will be started at next input file transition.");
    desc.addUntracked<int>("compressionLevel", 7)
        ->setComment("ROOT compression level of output file.");
    desc.addUntracked<std::string>("compressionAlgorithm", "ZLIB")
        ->setComment("Algorithm used to compress data in the ROOT output file, allowed values are ZLIB and LZMA");
    desc.addUntracked<int>("basketSize", 16384)
        ->setComment("Default ROOT basket size in output file.");
    desc.addUntracked<int>("eventAutoFlushCompressedSize",-1)->setComment("Set ROOT auto flush stored data size (in bytes) for event TTree. The value sets how large the compressed buffer is allowed to get. The uncompressed buffer can be quite a bit larger than this depending on the average compression ratio. The value of -1 just uses ROOT's default value. The value of 0 turns off this feature.");
    desc.addUntracked<int>("splitLevel", 99)
        ->setComment("Default ROOT branch split level in output file.");
    desc.addUntracked<std::string>("sortBaskets", std::string("sortbasketsbyoffset"))
        ->setComment("Legal values: 'sortbasketsbyoffset', 'sortbasketsbybranch', 'sortbasketsbyentry'.\n"
                     "Used by ROOT when fast copying. Affects performance.");
    desc.addUntracked<int>("treeMaxVirtualSize", -1)
        ->setComment("Size of ROOT TTree TBasket cache.  Affects performance.");
    desc.addUntracked<bool>("fastCloning", true)
        ->setComment("True:  Allow fast copying, if possible.\n"
                     "False: Disable fast copying.");
    desc.addUntracked<bool>("overrideInputFileSplitLevels", false)
        ->setComment("False: Use branch split levels and basket sizes from input file, if possible.\n"
                     "True:  Always use specified or default split levels and basket sizes.");
    desc.addUntracked<bool>("writeStatusFile", false)
        ->setComment("Write a status file. Intended for use by workflow management.");
    desc.addUntracked<std::string>("dropMetaData", defaultString)
        ->setComment("Determines handling of per product per event metadata.  Options are:\n"
                     "'NONE':    Keep all of it.\n"
                     "'DROPPED': Keep it for products produced in current process and all kept products. Drop it for dropped products produced in prior processes.\n"
                     "'PRIOR':   Keep it for products produced in current process. Drop it for products produced in prior processes.\n"
                     "'ALL':     Drop all of it.");
    {
      ParameterSetDescription dataSet;
      dataSet.setAllowAnything();
      desc.addUntracked<ParameterSetDescription>("dataset", dataSet)
      ->setComment("PSet is only used by Data Operations and not by this module.");
    }
    {
      ParameterSetDescription specialSplit;
      specialSplit.addUntracked<std::string>("branch")->setComment("Name of branch needing a special split level. The name can contain wildcards '*' and '?'");
      specialSplit.addUntracked<int>("splitLevel")->setComment("The special split level for the branch");
      desc.addVPSetUntracked("overrideBranchesSplitLevel",specialSplit, std::vector<ParameterSet>());
    }
  }
}
