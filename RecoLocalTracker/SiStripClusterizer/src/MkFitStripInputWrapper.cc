#include "RecoLocalTracker/SiStripClusterizer/interface/MkFitStripInputWrapper.h"

// mkFit includes
#include "Hit.h"
#include "LayerNumberConverter.h"
#include "Track.h"

MkFitStripInputWrapper::MkFitStripInputWrapper() = default;

MkFitStripInputWrapper::MkFitStripInputWrapper(//MkFitHitIndexMap hitIndexMap,
                                     std::vector<mkfit::HitVec> hits,
                                     //mkfit::TrackVec seeds,
                                     mkfit::LayerNumberConverter const& lnc)
    : //hitIndexMap_{std::move(hitIndexMap)},
      hits_{std::move(hits)},
      //seeds_{std::make_unique<mkfit::TrackVec>(std::move(seeds))},
      lnc_{std::make_unique<mkfit::LayerNumberConverter>(lnc)} {}

MkFitStripInputWrapper::~MkFitStripInputWrapper() = default;

MkFitStripInputWrapper::MkFitStripInputWrapper(MkFitStripInputWrapper&&) = default;
MkFitStripInputWrapper& MkFitStripInputWrapper::operator=(MkFitStripInputWrapper&&) = default;

unsigned int MkFitStripInputWrapper::nlayers() const { return lnc_->nLayers(); }
