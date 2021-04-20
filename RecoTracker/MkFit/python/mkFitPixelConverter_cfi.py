import FWCore.ParameterSet.Config as cms

from RecoTracker.MkFit.mkFitPixelConverterDefault_cfi import mkFitPixelConverterDefault as _mkFitPixelConverterDefault
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

mkFitPixelConverter = _mkFitPixelConverterDefault.clone(
    minGoodStripCharge = cms.PSet(
        refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
