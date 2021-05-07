#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitRecHitWrapper.h"

// mkFit includes
#include "Hit.h"

MkFitRecHitWrapper::MkFitRecHitWrapper() = default;

MkFitRecHitWrapper::MkFitRecHitWrapper(std::vector<mkfit::HitVec> hits,
                                       int totalHits,
                                       std::vector<std::vector<float>> barycenters,
                                       std::vector<std::vector<int>> detIds)
    : hits_{std::move(hits)}, totalHits_{totalHits}, barycenters_{std::move(barycenters)}, detIds_{std::move(detIds)} {}

MkFitRecHitWrapper::~MkFitRecHitWrapper() = default;

MkFitRecHitWrapper::MkFitRecHitWrapper(MkFitRecHitWrapper&&) = default;
MkFitRecHitWrapper& MkFitRecHitWrapper::operator=(MkFitRecHitWrapper&&) = default;
