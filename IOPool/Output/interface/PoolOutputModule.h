#ifndef IOPool_Output_PoolOutputModule_h
#define IOPool_Output_PoolOutputModule_h

//////////////////////////////////////////////////////////////////////
//
// Class PoolOutputModule. Output module to POOL file
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include "IOPool/Output/interface/PoolOutputModuleBase.h"
#include "FWCore/Framework/interface/one/OutputModule.h"

class TTree;
namespace edm {

  class PoolOutputModule : public one::OutputModule<WatchInputFiles>, public PoolOutputModuleBase {
  public:
    explicit PoolOutputModule(ParameterSet const& ps);
    ~PoolOutputModule() override;
    PoolOutputModule(PoolOutputModule const&) = delete; // Disallow copying and moving
    PoolOutputModule& operator=(PoolOutputModule const&) = delete; // Disallow copying and moving

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
    void setProcessesWithSelectedMergeableRunProducts(std::set<std::string> const&) override;

  private:
    void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const override;

    void openFile(FileBlock const& fb) override;
    void respondToOpenInputFile(FileBlock const& fb) override;
    void respondToCloseInputFile(FileBlock const& fb) override;
    void writeLuminosityBlock(LuminosityBlockForOutput const& lb) override;
    void writeRun(RunForOutput const& r) override;
    bool isFileOpen() const override;
    void reallyOpenFile();
    void reallyCloseFile() override;
    void beginJob() override;

    void beginInputFile(FileBlock const& fb);

    edm::propagate_const<std::unique_ptr<RootOutputFile>> rootOutputFile_;
  };
}

#endif

