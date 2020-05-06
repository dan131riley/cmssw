#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripConditionsGPUWrapper.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "SiStripRawToClusterGPUKernel.h"

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

    clust_data_d = std::make_unique<clust_data_t>();
    clust_data = std::make_unique<clust_data_t>();

    sst_data_d->nStrips = max_strips;

    chanlocsGPU = std::make_unique<ChannelLocsGPU>(detmap.size(), stream);
    chanlocsGPU->setvals(chanlocs.get(), inputGPU, stream);

    stripdata = std::make_unique<StripDataGPU>(max_strips, stream);
    const int max_seedstrips = MAX_SEEDSTRIPS;

    auto condGPU = conditionswrapper->getGPUProductAsync(stream);

    unpackChannelsGPU(chanlocsGPU.get(), condGPU, stripdata.get(), stream);

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

    allocateSSTDataGPU(max_strips, stripdata.get(), sst_data_d.get(), &pt_sst_data_d, stream);

    setSeedStripsNCIndexGPU(sst_data_d.get(), pt_sst_data_d, condGPU, stream);

    allocateClustDataGPU(max_seedstrips, clust_data_d.get(), &pt_clust_data_d, stream);
    allocateClustData(max_seedstrips, clust_data.get(), stream);

    findClusterGPU(sst_data_d.get(), pt_sst_data_d, condGPU,
                   clust_data.get(), clust_data_d.get(), pt_clust_data_d,
                   stream);
  }

  void SiStripRawToClusterGPUKernel::copyAsync(cudaStream_t stream)
  {
    cpyGPUToCPU(clust_data.get(), clust_data_d.get(), stream);
  }

  std::unique_ptr<edmNew::DetSetVector<SiStripCluster>>
  SiStripRawToClusterGPUKernel::getResults(cudaStream_t stream)
  {
    using out_t = edmNew::DetSetVector<SiStripCluster>;

    std::unique_ptr<out_t> output(new edmNew::DetSetVector<SiStripCluster>());

    const int nSeedStripsNC = clust_data->nSeedStripsNC;
    const auto clusterSize = clust_data->clusterSize;
    const auto ADCs = clust_data->clusterADCs;
    const auto detIDs = clust_data->clusterDetId;
    const auto stripIDs = clust_data->firstStrip;
    const auto trueCluster = clust_data->trueCluster;

    output->reserve(15000, nSeedStripsNC);

    std::vector<uint8_t> adcs;
    adcs.reserve(kClusterMaxStrips);

    for (int i = 0; i < nSeedStripsNC;) {
      const auto detid = detIDs[i];
      out_t::FastFiller record(*output, detid);

      while (i < nSeedStripsNC && detIDs[i] == detid) {
        if (trueCluster[i]) {
          const auto size = std::min(clusterSize[i], kClusterMaxStrips);
          const auto firstStrip = stripIDs[i];

          adcs.clear();
          for (auto j = 0; j < size; ++j) {
            adcs.push_back(ADCs[i+j*nSeedStripsNC]);
          }
          record.push_back(SiStripCluster(firstStrip, adcs.begin(), adcs.end()));
        }
        i++;
      }

//#define DSRDEBUG
#ifdef DSRDEBUG
      if (detid == 369120277) {
        std::cout << "Printing clusters for detid " << detid << std::endl;
        for (const auto& cluster : record) {
          std::cout << "Cluster " << cluster.firstStrip() << ": ";
          for (const auto& ampl : cluster.amplitudes()) {
            std::cout << (int) ampl << " ";
          }
          std::cout << std::endl;
        }
      }
#endif
    }

    output->shrink_to_fit();

    freeClustDataGPU(clust_data_d.get(), pt_clust_data_d, stream);
    freeSSTDataGPU(sst_data_d.get(), pt_sst_data_d, stream);
    freeClustData(clust_data.get());
    reset();

    return output;
  }

  void SiStripRawToClusterGPUKernel::reset()
  {
    fedRawDataGPU.reset();
    chanlocs.reset();
    chanlocsGPU.reset();
    stripdata.reset();
    sst_data_d.reset();
    clust_data_d.reset();
    clust_data.reset();
  }
}

