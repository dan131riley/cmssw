import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.cmsIdealGeometry2015devXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.ForwardGeometry.zdcTopologyEP_cfi import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *
from Geometry.MuonNumbering.muonOffsetESProducer_cff import *

