#ifndef RecoLocalTracker_Records_SiStripGPULocalToGlobalMap_h
#define RecoLocalTracker_Records_SiStripGPULocalToGlobalMap_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class SiStripGPULocalToGlobalMapRcd
    : public edm::eventsetup::DependentRecordImplementation<SiStripGPULocalToGlobalMapRcd,
                                                            edm::mpl::Vector<IdealGeometryRecord,
                                                                             IdealMagneticFieldRecord,
                                                                             SiStripBackPlaneCorrectionDepRcd,
                                                                             SiStripLorentzAngleRcd,
                                                                             TrackerDigiGeometryRecord>> {};

#endif  // RecoLocalTracker_Records_SiStripGPULocalToGlobalMap_h
