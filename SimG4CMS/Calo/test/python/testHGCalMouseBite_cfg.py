import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "V18",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: V16, V17, V17Shift, V18, V19")

### get and parse the command line arguments
options.parseArguments()
print(options)


####################################################################
# Use the options
if (options.type == "V18"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process("HGCalMouseBiteTest",Phase2C22I13M9)
    geomFile = "Configuration.Geometry.GeometryExtendedRun4D104_cff"
elif (options.type == "V19"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process("HGCalMouseBiteTest",Phase2C22I13M9)
    geomFile = "Configuration.Geometry.GeometryExtendedRun4D120_cff"
elif (options.type == "V17Shift"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process("HGCalMouseBiteTest",Phase2C17I13M9)
    geomFile = "Geometry.HGCalCommonData.testHGCal" + options.type + "Reco_cff"
elif (options.type == "V16"):
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process("HGCalMouseBiteTest",Phase2C17I13M9)
    geomFile = "Configuration.Geometry.GeometryExtendedRun4D100_cff"
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process("HGCalMouseBiteTest",Phase2C17I13M9)
    geomFile = "Configuration.Geometry.GeometryExtendedRun4D110_cff"

print("Geometry file: ", geomFile)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalGeom=dict()
    process.MessageLogger.HGCSim=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
                                   PGunParameters = cms.PSet(
                                       PartID = cms.vint32(14),
                                       MinEta = cms.double(-3.5),
                                       MaxEta = cms.double(3.5),
                                       MinPhi = cms.double(-3.14159265359),
                                       MaxPhi = cms.double(3.14159265359),
                                       MinE   = cms.double(9.99),
                                       MaxE   = cms.double(10.01)
                                   ),
                                   AddAntiParticle = cms.bool(False),
                                   Verbosity       = cms.untracked.int32(0),
                                   firstRun        = cms.untracked.uint32(1)
                               )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.load("SimG4CMS.Calo.hgcalMouseBiteTester_cfi")

 
process.p1 = cms.Path(process.generator*process.hgcalMouseBiteTester)
