import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.mkFitRecHitsMapDefault_cfi import mkFitRecHitsMapDefault as _mkFitRecHitsMap

mkFitRecHitsMapConverter = _mkFitRecHitsMap.clone(
)

