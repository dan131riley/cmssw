import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.mkFitHitsFromSOADefault_cfi import mkFitHitsFromSOADefault as _mkFitHits

mkFitHitConverter = _mkFitHits.clone(
    #minGoodStripCharge = cms.PSet(
    #    refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    # productLabel= cms.InputTag('siStripRawToSOAClusters')
)

