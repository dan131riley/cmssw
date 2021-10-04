#ifndef RecoLocalTracker_Records_SiStripClusterizerConditionsGPURcd_h
#define RecoLocalTracker_Records_SiStripClusterizerConditionsGPURcd_h
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripClusterizerConditionsGPURcd : public edm::eventsetup::DependentRecordImplementation<
                                               SiStripClusterizerConditionsGPURcd,
                                               edm::mpl::Vector<SiStripGainRcd, SiStripNoisesRcd, SiStripQualityRcd>> {
};

#endif  // RecoLocalTracker_Records_SiStripClusterizerConditionsGPURcd_h
