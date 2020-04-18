#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripConditionsGPUWrapper.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "SiStripRawToClusterGPUKernel.h"
#include "ChanLocsGPU.h"
#include "clusterGPU.cuh"

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
    auto fedRawDataGPU = cms::cuda::make_device_unique<uint8_t[]>(totalSize, stream);

    size_t off = 0;
    std::vector<stripgpu::fedId_t> fedIndex(stripgpu::kFedCount, stripgpu::invFed);
    std::vector<stripgpu::fedId_t> fedIdv;
    std::vector<size_t> fedRawDataOffsets;

    fedRawDataOffsets.reserve(stripgpu::kFedCount);
    fedIdv.reserve(stripgpu::kFedCount);

    sistrip::FEDReadoutMode mode = sistrip::READOUT_MODE_INVALID;

    for (size_t fedi = 0; fedi < buffers.size(); ++fedi) {
      auto& buff = buffers[fedi];
      if (buff != nullptr) {
        const auto raw = rawdata[fedi];
        memcpy(fedRawDataHost.get() + off, raw->data(), raw->size());
        fedIndex[stripgpu::fedIndex(fedi)] = fedIdv.size();
        fedIdv.push_back(fedi);
        fedRawDataOffsets.push_back(off);
        off += raw->size();
        if (fedIdv.size() == 1) {
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
    ChannelLocs chanlocs(detmap.size(), stream);
    ChannelLocsGPU chanlocsGPU(detmap.size(), stream);
    std::vector<uint8_t*> inputGPU(chanlocs.size());

    // iterate over the detector in DetID/APVPair order
    // mapping out where the data are
    for(size_t i = 0; i < detmap.size(); ++i) {
      const auto& detp = detmap[i];

      auto fedId = detp.fedID();
      auto fedi = fedIndex[stripgpu::fedIndex(fedId)];
      if (fedi != invFed) {
        const auto buffer = buffers[fedId].get();
        const auto& channel = buffer->channel(detp.fedCh());

        if (channel.length() >= headerlen) {
          chanlocs.setChannelLoc(i, channel.data(), channel.offset()+headerlen, offset, channel.length()-headerlen,
                                 detp.fedID(), detp.fedCh());
          inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets[fedi] + (channel.data() - rawdata[fedId]->data());
          offset += channel.length()-headerlen;
        } else {
          chanlocs.setChannelLoc(i, channel.data(), channel.offset(), offset, channel.length(),
                                 detp.fedID(), detp.fedCh());
          inputGPU[i] = fedRawDataGPU.get() + fedRawDataOffsets[fedi] + (channel.data() - rawdata[fedId]->data());
          offset += channel.length();
          assert(channel.length() == 0);
        }
      } else {
        chanlocs.setChannelLoc(i, nullptr, 0, 0, 0, invFed, 0);
        inputGPU[i] = nullptr;
      }
    }

    const auto max_strips = offset;

    auto sst_data_d = std::make_unique<sst_data_t>();
    sst_data_t *pt_sst_data_d;

    auto clust_data_d = std::make_unique<clust_data_t>();
    auto clust_data = std::make_unique<clust_data_t>();
    clust_data_t *pt_clust_data_d;

    sst_data_d->nStrips = max_strips;

    chanlocsGPU.reset(chanlocs, inputGPU, stream);
    StripDataGPU stripdata(max_strips, stream);
    const int max_seedstrips = MAX_SEEDSTRIPS;

    auto condGPU = conditionswrapper->getGPUProductAsync(stream);

    unpackChannelsGPU(chanlocsGPU, condGPU, stripdata, stream);

    //#define VERIFY
#ifdef VERIFY
    auto outdata = cms::cuda::make_host_unique<uint8_t[]>(max_strips, stream);
    auto stripid = cms::cuda::make_host_unique<stripgpu::stripId_t[]>(max_strips, stream);
    cms::cuda::copyAsync(outdata, stripdata.alldataGPU_, max_strips, stream);
    cms::cuda::copyAsync(stripid, stripdata.stripIdGPU_, max_strips, stream);
    cudaCheck(cudaStreamSynchronize(stream));

    for(size_t i = 0; i < chanlocs.size(); ++i) {
      const auto data = chanlocs.input(i);

      if (data != nullptr) {
        auto aoff = chanlocs.offset(i);
        auto choff = chanlocs.inoff(i);

        for (auto k = 0; k < chanlocs.length(i); ++k) {
          assert(data[choff^7] == outdata[aoff]);
          aoff++; choff++;
          std::cout << "strip id " << stripid[aoff] << " adc " << (uint32_t) outdata[aoff] << std::endl;
        }
      }
    }
    outdata.reset(nullptr);
#endif

    allocateSSTDataGPU(max_strips, stripdata, sst_data_d.get(), &pt_sst_data_d, stream);

    setSeedStripsNCIndexGPU(sst_data_d.get(), pt_sst_data_d, condGPU, stream);

    allocateClustDataGPU(max_seedstrips, clust_data_d.get(), &pt_clust_data_d, stream);

    findClusterGPU(sst_data_d.get(), pt_sst_data_d,
                   condGPU,
                   clust_data_d.get(), pt_clust_data_d,
                   stream);

    allocateClustData(max_seedstrips, clust_data.get(), stream);

    cpyGPUToCPU(sst_data_d.get(), pt_sst_data_d,
                clust_data.get(), clust_data_d.get(),
                stream);

    freeClustDataGPU(clust_data_d.get(), pt_clust_data_d, stream);
    freeSSTDataGPU(sst_data_d.get(), pt_sst_data_d, stream);
    freeClustData(clust_data.get());
  }

  void SiStripRawToClusterGPUKernel::getResults()
  {

  }
}

