#ifndef RecoTracker_MkFit_MkFitRecHitWrapper_h
#define RecoTracker_MkFit_MkFitRecHitWrapper_h

#include "RecoTracker/MkFit/interface/MkFitHitIndexMap.h"
#include "CUDADataFormats/SiStripCluster/interface/GPUtypes.h"

#include <vector>

namespace mkfit {
  class Hit;
  class LayerNumberConverter;
  using HitVec = std::vector<Hit>;
}  // namespace mkfit

class MkFitRecHitWrapper {
public:
  MkFitRecHitWrapper();
  MkFitRecHitWrapper(int totalHits,
                     std::vector<mkfit::HitVec> hits,
                     std::vector<std::vector<stripgpu::stripId_t>> firstStrips,
                     std::vector<std::vector<stripgpu::detId_t>> detIds);
  ~MkFitRecHitWrapper();

  MkFitRecHitWrapper(MkFitRecHitWrapper const&) = delete;
  MkFitRecHitWrapper& operator=(MkFitRecHitWrapper const&) = delete;
  MkFitRecHitWrapper(MkFitRecHitWrapper&&);
  MkFitRecHitWrapper& operator=(MkFitRecHitWrapper&&);

  std::vector<mkfit::HitVec> const& hits() const { return hits_; }
  int totalHits() const { return totalHits_; }
  std::vector<std::vector<stripgpu::stripId_t>> const& firstStrips() const { return firstStrips_; }
  std::vector<std::vector<stripgpu::detId_t>> const& detIds() const { return detIds_; }

private:
  int totalHits_;
  std::vector<mkfit::HitVec> hits_;
  std::vector<std::vector<stripgpu::stripId_t>> firstStrips_;
  std::vector<std::vector<stripgpu::detId_t>> detIds_;
};

#endif
