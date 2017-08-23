#ifndef IOPool_Output_PoolOutputModuleBase_h
#define IOPool_Output_PoolOutputModuleBase_h

//////////////////////////////////////////////////////////////////////
//
// Base Class for PoolOutputModules. Output module to POOL file
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <array>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <regex>

#include "IOPool/Common/interface/RootServiceChecker.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Framework/interface/OutputModule.h"

#include "tbb/concurrent_unordered_map.h"
#include "tbb/concurrent_unordered_set.h"

class TTree;
namespace edm {

  class EDGetToken;
  class ModuleCallingContext;
  class ParameterSet;
  class RootOutputFile;
  class ConfigurationDescriptions;
  class ProductProvenanceRetriever;
  class ThinnedAssociationsHelper;
  class SubProcessParentageHelper;

  class PoolOutputModuleBase {
  public:
    enum DropMetaData { DropNone, DropDroppedPrior, DropPrior, DropAll };
    explicit PoolOutputModuleBase(ParameterSet const& ps, bool wantAllEvents);
    virtual ~PoolOutputModuleBase() noexcept (false);
    PoolOutputModuleBase(PoolOutputModuleBase const&) = delete; // Disallow copying and moving
    PoolOutputModuleBase& operator=(PoolOutputModuleBase const&) = delete; // Disallow copying and moving
    std::string const& fileName() const {return fileName_;}
    std::string const& logicalFileName() const {return logicalFileName_;}
    int const compressionLevel() const {return compressionLevel_;}
    std::string const& compressionAlgorithm() const {return compressionAlgorithm_;}
    int const basketSize() const {return basketSize_;}
    int eventAutoFlushSize() const {return eventAutoFlushSize_;}
    int const splitLevel() const {return splitLevel_;}
    std::string const& basketOrder() const {return basketOrder_;}
    int const treeMaxVirtualSize() const {return treeMaxVirtualSize_;}
    bool const overrideInputFileSplitLevels() const {return overrideInputFileSplitLevels_;}
    DropMetaData const dropMetaData() const {return dropMetaData_;}
    std::string const& catalog() const {return catalog_;}
    std::string const& moduleLabel() const {return moduleLabel_;}
    unsigned int const maxFileSize() const {return maxFileSize_;}
    int const inputFileCount() const {return inputFileCount_;}
    int const whyNotFastClonable() const {return whyNotFastClonable_;}

    static void fillDescription(ParameterSetDescription& desc);

    struct AuxItem {
      AuxItem();
      ~AuxItem() {}
      int basketSize_;
    };
    typedef std::array<AuxItem, NumBranchTypes> AuxItemArray;
    AuxItemArray const& auxItems() const {return auxItems_;}

    struct OutputItem {
      class Sorter {
      public:
        explicit Sorter(TTree* tree);
        bool operator() (OutputItem const& lh, OutputItem const& rh) const;
      private:
        std::shared_ptr<std::map<std::string, int> > treeMap_;
      };

      OutputItem();

      explicit OutputItem(BranchDescription const* bd, EDGetToken const& token, int splitLevel, int basketSize);

      //user destructor disables generation of a default move assignment operator
      //~OutputItem() {}

      BranchID branchID() const { return branchDescription_->branchID(); }
      std::string const& branchName() const { return branchDescription_->branchName(); }

      bool operator <(OutputItem const& rh) const {
        return *branchDescription_ < *rh.branchDescription_;
      }

      BranchDescription const* branchDescription_;
      EDGetToken token_;
      //mutable void const* product_; // moved to RootOutputFile, MUST be kept in sync there.
      int splitLevel_;
      int basketSize_;
    };

    typedef std::vector<OutputItem> OutputItemList;

    typedef std::array<OutputItemList, NumBranchTypes> OutputItemListArray;

    struct SpecialSplitLevelForBranch {
      SpecialSplitLevelForBranch(std::string const& iBranchName, int iSplitLevel):
      branch_(convert(iBranchName)),
      splitLevel_(iSplitLevel < 1? 1: iSplitLevel) //minimum is 1
      {}
      bool match(std::string const& iBranchName) const;
      std::regex convert(std::string const& iGlobBranchExpression )const;
      
      std::regex branch_;
      int splitLevel_;
    };
    
    OutputItemListArray const& selectedOutputItemList() const {return selectedOutputItemList_;}

    BranchChildren const& branchChildren() const {return branchChildren_;}

    // these must be forwarded by the OutputModule implementation
    virtual bool OMwantAllEvents() const = 0;
    virtual BranchIDLists const* OMbranchIDLists() = 0;
    virtual ThinnedAssociationsHelper const* OMthinnedAssociationsHelper() const = 0;
    virtual ParameterSetID OMselectorConfig() const = 0;
    virtual SelectedProductsForBranchType const& OMkeptProducts() const = 0;

  protected:
    virtual std::pair<std::string, std::string> physicalAndLogicalNameForNewFile();
    virtual void doExtrasAfterCloseFile();
    virtual void beginJobBase();
    virtual void respondToOpenInputFileBase(FileBlock const& fb, SelectedProductsForBranchType const& keptProducts);
    virtual void reallyCloseFileBase(RootOutputFile* rootOutputFile, bool doWrite = false);
    virtual void preActionBeforeRunEventAsyncBase(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext,
                                                  Principal const& iPrincipal) const;
    void beginInputFileBase(FileBlock const& fb);
    void updateBranchParents(EventForOutput const& e, SubProcessParentageHelper const* helper);
    std::string const& statusFileName() const {return statusFileName_;}
    bool initializedFromInput() { return initializedFromInput_; }

  private:
    struct BranchIDhash {
      BranchIDhash() {}
      size_t operator()(const BranchID& b) const
      {
        return tbb::tbb_hasher(b.id());
      }
    };
    struct ParentageIDhash {
      ParentageIDhash() {}
      size_t operator()(const ParentageID& b) const
      {
        return b.smallHash();
      }
    };


    typedef tbb::concurrent_unordered_set<ParentageID, ParentageIDhash> ParentSet;
    typedef tbb::concurrent_unordered_map<BranchID, ParentSet, BranchIDhash> BranchParents;
    void updateBranchParentsForOneBranch(ProductProvenanceRetriever const* provRetriever,
                                         BranchID const& branchID);
    void fillDependencyGraph();

    void startEndFile(RootOutputFile* rootOutputFile);
    void writeFileFormatVersion(RootOutputFile* rootOutputFile);
    void writeFileIdentifier(RootOutputFile* rootOutputFile);
    void writeIndexIntoFile(RootOutputFile* rootOutputFile);
    void writeProcessHistoryRegistry(RootOutputFile* rootOutputFile);
    void writeParameterSetRegistry(RootOutputFile* rootOutputFile);
    void writeProductDescriptionRegistry(RootOutputFile* rootOutputFile);
    void writeParentageRegistry(RootOutputFile* rootOutputFile);
    void writeBranchIDListRegistry(RootOutputFile* rootOutputFile);
    void writeThinnedAssociationsHelper(RootOutputFile* rootOutputFile);
    void writeProductDependencies(RootOutputFile* rootOutputFile);
    void finishEndFile(RootOutputFile* rootOutputFile, bool doWrite = false);

    void fillSelectedItemList(BranchType branchtype, TTree* theInputTree, SelectedProductsForBranchType const& keptProducts);

    RootServiceChecker rootServiceChecker_;
    AuxItemArray auxItems_;
    OutputItemListArray selectedOutputItemList_;
    std::vector<SpecialSplitLevelForBranch> specialSplitLevelForBranches_;
    std::string const fileName_;
    std::string const logicalFileName_;
    std::string const catalog_;
    unsigned int const maxFileSize_;
    int const compressionLevel_;
    std::string const compressionAlgorithm_;
    int const basketSize_;
    int const eventAutoFlushSize_;
    int const splitLevel_;
    std::string basketOrder_;
    int const treeMaxVirtualSize_;
    int whyNotFastClonable_;
    DropMetaData dropMetaData_;
    std::string const moduleLabel_;
    bool initializedFromInput_;
    int outputFileCount_;
    int inputFileCount_;
    unsigned int childIndex_;
    unsigned int numberOfDigitsInIndex_;
    BranchParents branchParents_;
    BranchChildren branchChildren_;
    std::vector<BranchID> producedBranches_;
    bool overrideInputFileSplitLevels_;
    std::string statusFileName_;
  };
}

#endif

