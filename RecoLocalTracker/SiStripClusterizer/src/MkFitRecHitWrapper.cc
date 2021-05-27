#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitRecHitWrapper.h"

// mkFit includes
#include "Hit.h"

MkFitRecHitWrapper::MkFitRecHitWrapper() = default;

MkFitRecHitWrapper::MkFitRecHitWrapper(int totalHits,
                                       std::vector<mkfit::HitVec> hits,
                                       std::vector<std::vector<stripgpu::stripId_t>> firstStrips,
                                       std::vector<std::vector<stripgpu::detId_t>> detIds)
    : totalHits_{totalHits}, hits_{hits}, firstStrips_{firstStrips}, detIds_{detIds} {}

MkFitRecHitWrapper::~MkFitRecHitWrapper() = default;

MkFitRecHitWrapper::MkFitRecHitWrapper(MkFitRecHitWrapper&&) = default;
MkFitRecHitWrapper& MkFitRecHitWrapper::operator=(MkFitRecHitWrapper&&) = default;
