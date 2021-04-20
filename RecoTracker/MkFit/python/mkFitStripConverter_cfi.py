import FWCore.ParameterSet.Config as cms

from RecoTracker.MkFit.mkFitStripConverterDefault_cfi import mkFitStripConverterDefault as _mkFitStripConverterDefault
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

mkFitStripConverter = _mkFitStripConverterDefault.clone(
    minGoodStripCharge = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
