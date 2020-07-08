#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripConditionsGPUWrapper.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "SiStripRawToClusterGPUKernel.h"

#include "ChanLocsGPU.h"

namespace stripgpu {
  void SiStripRawToClusterGPUKernel::makeAsync(
                   const std::vector<const FEDRawData*>& rawdata,
                   const std::vector<std::unique_ptr<sistrip::FEDBuffer>>& buffers,
                   const SiStripConditionsGPUWrapper* conditionswrapper,
                   cudaStream_t stream) {

    size_t totalSize{0};
    for (const auto& buff : buffers) {
      if (buff != nullptr) {
        totalSize += buff->bufferSize();
      }
    }

    auto fedRawDataHost = cms::cuda::make_host_unique<uint8_t[]>(totalSize, stream);
    fedRawDataGPU = cms::cuda::make_device_unique<uint8_t[]>(totalSize, stream);

    size_t off = 0;
    fedRawDataOffsets.clear();
    fedIndex.clear();
    fedIndex.resize(stripgpu::kFedCount, stripgpu::invFed);

    sistrip::FEDReadoutMode mode = sistrip::READOUT_MODE_INVALID;

    for (size_t fedi = 0; fedi < buffers.size(); ++fedi) {
      auto& buff = buffers[fedi];
      if (buff != nullptr) {
        const auto raw = rawdata[fedi];
        memcpy(fedRawDataHost.get() + off, raw->data(), raw->size());
        fedIndex[stripgpu::fedIndex(fedi)] = fedRawDataOffsets.size();
        fedRawDataOffsets.push_back(off);
        off += raw->size();
        if (fedRawDataOffsets.size() == 1) {
          mode = buff->readoutMode();
        } else {
          assert(buff->readoutMode() == mode);
        }
      
      }
    }
    // send rawdata to GPU
    cms::cuda::copyAsync(fedRawDataGPU, fedRawDataHost, totalSize, stream);

    const auto& detmap = conditionswrapper->detToFeds();
    const uint16_t headerlen = mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED ? 7 : 2;
    size_t offset = 0;
    chanlocs = std::make_unique<ChannelLocs>(detmap.size(), stream);
    std::vector<uint8_t*> inputGPU(chanlocs->size());

    // iterate over the detector in DetID/APVPair order
    // mapping out where the data are
    for(size_t i = 0; i < detmap.size(); ++i) {
      const auto& detp = detmap[i];
      const auto fedId = detp.fedID();
      const auto fedCh = detp.fedCh();
      const auto fedi = fedIndex[stripgpu::fedIndex(fedId)];

      if (fedi != invFed) {
        const auto buffer = buffers[fedId].get();
        const auto& channel = buffer->channel(detp.fedCh());

        if (channel.length() >= headerlen) {
          auto len = channel.length() - headerlen;
          chanlocs->setChannelLoc(i, channel.data(), channel.offset()+headerlen, offset, len,
                                 fedId, fedCh);
          inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets[fedi] + (channel.data() - rawdata[fedId]->data());
          offset += len;
        } else {
          chanlocs->setChannelLoc(i, channel.data(), channel.offset(), offset, channel.length(),
                                 fedId, fedCh);
          inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets[fedi] + (channel.data() - rawdata[fedId]->data());
          offset += channel.length();
          assert(channel.length() == 0);
        }
      } else {
        chanlocs->setChannelLoc(i, nullptr, 0, 0, 0, invFed, 0);
        inputGPU[i] = nullptr;
      }
    }

    const auto max_strips = offset;

    sst_data_d = cms::cuda::make_host_unique<sst_data_t>(stream);
    sst_data_d->nStrips = max_strips;

    chanlocsGPU = std::make_unique<ChannelLocsGPU>(detmap.size(), stream);
    chanlocsGPU->setvals(chanlocs.get(), inputGPU, stream);

    stripdata = std::make_unique<StripDataGPU>(max_strips, stream);
    const int max_seedstrips = MAX_SEEDSTRIPS;

    auto condGPU = conditionswrapper->getGPUProductAsync(stream);

    unpackChannelsGPU(condGPU, stream);

//#define VERIFY
#ifdef VERIFY
    auto outdata = cms::cuda::make_host_unique<uint8_t[]>(max_strips, stream);
    cms::cuda::copyAsync(outdata, stripdata->alldataGPU_, max_strips, stream);
    cudaCheck(cudaStreamSynchronize(stream));

    for(size_t i = 0; i < chanlocs->size(); ++i) {
      const auto data = chanlocs->input(i);
      const auto len = chanlocs->length(i);

      if (data != nullptr && len > 0) {
        auto aoff = chanlocs->offset(i);
        auto choff = chanlocs->inoff(i);
        const auto end = choff + len;

        while (choff < end) {
          const auto stripIndex = data[choff++^7];
          const auto groupLength = data[choff++^7];
          aoff += 2;
          for (auto k = 0; k < groupLength; ++k, ++choff, ++aoff) {
            if (data[choff^7] != outdata[aoff]) {
              std::cout << "i:k " << i << ":" << k << " " << (uint32_t) data[choff^7] << " != " << (uint32_t) outdata[aoff] << std::endl;
            }
          }
        }
      }
    }
    outdata.reset(nullptr);
#endif

    allocateSSTDataGPU(max_strips, stream);

    setSeedStripsNCIndexGPU(condGPU, stream);

    clusters_d = SiStripClustersCUDA(max_seedstrips, kClusterMaxStrips, stream);
    findClusterGPU(condGPU, stream);
  }

  SiStripClustersCUDA
  SiStripRawToClusterGPUKernel::getResults(cudaStream_t stream)
  {
    freeSSTDataGPU(stream);
    reset();

    return std::move(clusters_d);
  }

  void SiStripRawToClusterGPUKernel::reset()
  {
    fedRawDataGPU.reset();
    chanlocs.reset();
    chanlocsGPU.reset();
    stripdata.reset();
    sst_data_d.reset();
  }
}

