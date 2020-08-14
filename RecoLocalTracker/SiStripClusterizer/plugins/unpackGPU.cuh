#pragma once
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

class SiStripConditionsGPU;
class ChannelLocsGPU;
class StripDataGPU;

void unpackChannelsGPU(const ChannelLocsGPU* chanlocs, const SiStripConditionsGPU* conditions, StripDataGPU* stripdata, cudaStream_t stream);
