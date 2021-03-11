#ifndef RecoLocalTracker_Records_SiStripClusterizerGPUConditionsRcd_h
#define RecoLocalTracker_Records_SiStripClusterizerGPUConditionsRcd_h
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"

#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripClusterizerGPUConditionsRcd : public edm::eventsetup::DependentRecordImplementation<
                                               SiStripClusterizerGPUConditionsRcd,
                                               edm::mpl::Vector<SiStripGainRcd, SiStripNoisesRcd, SiStripQualityRcd>> {
};

#endif  // RecoLocalTracker_Records_SiStripClusterizerGPUConditionsRcd_h
