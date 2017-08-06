#ifndef IOPool_Output_ParallelPoolOutputModule_h
#define IOPool_Output_ParallelPoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// Class ParallelPoolOutputModule. Output module to POOL file
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////
#include "IOPool/Output/interface/PoolOutputModuleBase.h"
#include "FWCore/Framework/interface/global/OutputModule.h"

#include "tbb/concurrent_queue.h"

class TTree;
namespace ROOT {
  class TBufferMerger;
}

namespace edm {

  class ParallelPoolOutputModule : public global::OutputModule<WatchInputFiles>, public PoolOutputModuleBase {
  public:
    explicit ParallelPoolOutputModule(ParameterSet const& ps);
    virtual ~ParallelPoolOutputModule();
    ParallelPoolOutputModule(ParallelPoolOutputModule const&) = delete; // Disallow copying and moving
    ParallelPoolOutputModule& operator=(ParallelPoolOutputModule const&) = delete; // Disallow copying and moving

    std::string const& currentFileName() const;

    static void fillDescription(ParameterSetDescription& desc);
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

    using OutputModule::selectorConfig;

    // these must be forwarded by the OutputModule implementation
    virtual bool OMwantAllEvents() const override;
    virtual BranchIDLists const* OMbranchIDLists() override;
    virtual ThinnedAssociationsHelper const* OMthinnedAssociationsHelper() const override;
    virtual ParameterSetID OMselectorConfig() const override;
    virtual SelectedProductsForBranchType const& OMkeptProducts() const override;

  protected:
    ///allow inheriting classes to override but still be able to call this method in the overridden version
    virtual bool shouldWeCloseFile() const override;
    virtual void write(EventForOutput const& e) override;

  private:
    //virtual void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const override;

    virtual void openFile(FileBlock const& fb) override;
    virtual void respondToOpenInputFile(FileBlock const& fb) override;
    virtual void respondToCloseInputFile(FileBlock const& fb) override;
    virtual void writeLuminosityBlock(LuminosityBlockForOutput const& lb) override;
    virtual void writeRun(RunForOutput const& r) override;
    virtual void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) override;
    virtual bool isFileOpen() const override;
    virtual void reallyOpenFile() override;
    virtual void reallyCloseFile() override;
    virtual void beginJob() override;

    void beginInputFile(FileBlock const& fb);

    edm::propagate_const<std::shared_ptr<ROOT::TBufferMerger>> mergePtr_;
    edm::propagate_const<std::unique_ptr<RootOutputFile>> rootOutputFile_;

    // std::unique_ptr inside concurrent_bounded_queue doesn't instantiate??
    typedef tbb::concurrent_bounded_queue<RootOutputFile*> EventOutputFiles;
    EventOutputFiles eventOutputFiles_;
    std::atomic<unsigned int> eventFileCount_{};
    std::string moduleLabel_;
  };
}

#endif

