#ifndef RecoLocalTracker_SiStripClusterizer_MkFitStripInputWrapper_h
#define RecoLocalTracker_SiStripClusterizer_MkFitStripInputWrapper_h

#include "RecoTracker/MkFit/interface/MkFitHitIndexMap.h"

#include <memory>
#include <vector>

namespace mkfit {
  class Hit;
  class Track;
  class LayerNumberConverter;
  using HitVec = std::vector<Hit>;
//  using TrackVec = std::vector<Track>;
}  // namespace mkfit

class MkFitStripInputWrapper {
public:
  MkFitStripInputWrapper();
  MkFitStripInputWrapper(//MkFitHitIndexMap hitIndexMap,
                    std::vector<mkfit::HitVec> hits,
                    //mkfit::TrackVec seeds,
                    mkfit::LayerNumberConverter const& lnc);
  ~MkFitStripInputWrapper();

  MkFitStripInputWrapper(MkFitStripInputWrapper const&) = delete;
  MkFitStripInputWrapper& operator=(MkFitStripInputWrapper const&) = delete;
  MkFitStripInputWrapper(MkFitStripInputWrapper&&);
  MkFitStripInputWrapper& operator=(MkFitStripInputWrapper&&);

  //MkFitHitIndexMap const& hitIndexMap() const { return hitIndexMap_; }
  //mkfit::TrackVec const& seeds() const { return *seeds_; }
  std::vector<mkfit::HitVec> const& hits() const { return hits_; }
  mkfit::LayerNumberConverter const& layerNumberConverter() const { return *lnc_; }
  unsigned int nlayers() const;

private:
  //MkFitHitIndexMap hitIndexMap_;
  std::vector<mkfit::HitVec> hits_;
  //std::unique_ptr<mkfit::TrackVec> seeds_;            // for pimpl pattern
  std::unique_ptr<mkfit::LayerNumberConverter> lnc_;  // for pimpl pattern
};

#endif
