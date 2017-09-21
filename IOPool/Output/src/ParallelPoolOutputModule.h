#ifndef IOPool_Output_ParallelPoolOutputModule_h
#define IOPool_Output_ParallelPoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// Class ParallelPoolOutputModule. Parallel output to a ROOT file
//
// Author: Dan Riley
// Refactored from the PoolOutputModule by Luca Lista and Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <mutex>

#include "IOPool/Output/interface/PoolOutputModuleBase.h"
#include "FWCore/Framework/interface/limited/OutputModule.h"

#include "tbb/concurrent_queue.h"

class TTree;
namespace ROOT {
  namespace Experimental {
    class TBufferMerger;
  }
}

namespace edm {

  class ParallelPoolOutputModule : public limited::OutputModule<WatchInputFiles>, public PoolOutputModuleBase {
  public:
    explicit ParallelPoolOutputModule(ParameterSet const& ps);
    ~ParallelPoolOutputModule() override;
    ParallelPoolOutputModule(ParallelPoolOutputModule const&) = delete; // Disallow copying and moving
    ParallelPoolOutputModule& operator=(ParallelPoolOutputModule const&) = delete; // Disallow copying and moving

    std::string const& currentFileName() const;

    static void fillDescription(ParameterSetDescription& desc);
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

    using OutputModule::selectorConfig;

    // these must be forwarded by the OutputModule implementation
    bool OMwantAllEvents() const override;
    BranchIDLists const* OMbranchIDLists() override;
    ThinnedAssociationsHelper const* OMthinnedAssociationsHelper() const override;
    ParameterSetID OMselectorConfig() const override;
    SelectedProductsForBranchType const& OMkeptProducts() const override;

  protected:
    ///allow inheriting classes to override but still be able to call this method in the overridden version
    bool shouldWeCloseFile() const override;
    void write(EventForOutput const& e) override;

  private:
    void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const override;

    void openFile(FileBlock const& fb) override;
    void respondToOpenInputFile(FileBlock const& fb) override;
    void respondToCloseInputFile(FileBlock const& fb) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const& lb) override;
    void writeRun(RunForOutput const& r) override;
    bool isFileOpen() const override;
    void reallyCloseFile() override;
    void beginJob() override;

    void reallyOpenFile();
    void beginInputFile(FileBlock const& fb);

    edm::propagate_const<std::shared_ptr<ROOT::Experimental::TBufferMerger>> mergePtr_;
    edm::propagate_const<std::unique_ptr<RootOutputFile>> rootOutputFile_;

    // std::unique_ptr inside concurrent_bounded_queue doesn't instantiate??
    typedef tbb::concurrent_bounded_queue<std::shared_ptr<RootOutputFile>> EventOutputFiles;
    EventOutputFiles eventOutputFiles_;
    std::atomic<unsigned int> eventFileCount_{};
    std::string moduleLabel_;
    std::mutex notYetThreadSafe_;
  };
}

#endif

