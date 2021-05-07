#ifndef RecoTracker_MkFit_MkFitRecHitWrapper_h
#define RecoTracker_MkFit_MkFitRecHitWrapper_h

#include "RecoTracker/MkFit/interface/MkFitHitIndexMap.h"

#include <vector>

namespace mkfit {
  class Hit;
  class LayerNumberConverter;
  using HitVec = std::vector<Hit>;
}  // namespace mkfit

class MkFitRecHitWrapper {
public:
  MkFitRecHitWrapper();
  MkFitRecHitWrapper(std::vector<mkfit::HitVec> hits,
                     int totalHits,
                     std::vector<std::vector<float>> barycenters,
                     std::vector<std::vector<int>> detIds);
  ~MkFitRecHitWrapper();

  MkFitRecHitWrapper(MkFitRecHitWrapper const&) = delete;
  MkFitRecHitWrapper& operator=(MkFitRecHitWrapper const&) = delete;
  MkFitRecHitWrapper(MkFitRecHitWrapper&&);
  MkFitRecHitWrapper& operator=(MkFitRecHitWrapper&&);

  std::vector<mkfit::HitVec> const& hits() const { return hits_; }
  int const& totalHits() const { return totalHits_; }
  std::vector<std::vector<float>> const& barycenters() const { return barycenters_; }
  std::vector<std::vector<int>> const& detIds() const { return detIds_; }

private:
  std::vector<mkfit::HitVec> hits_;
  int totalHits_;
  std::vector<std::vector<float>> barycenters_;
  std::vector<std::vector<int>> detIds_;
};

#endif
