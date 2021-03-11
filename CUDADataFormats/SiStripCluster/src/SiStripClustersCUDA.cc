#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

SiStripClustersCUDA::SiStripClustersCUDA(size_t maxClusters, int clustersPerStrip, cudaStream_t stream) {
  clusterIndex_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
  clusterSize_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
  clusterADCs_d = cms::cuda::make_device_unique<uint8_t[]>(maxClusters*clustersPerStrip, stream);
  clusterDetId_d = cms::cuda::make_device_unique<stripgpu::detId_t[]>(maxClusters, stream);
  firstStrip_d = cms::cuda::make_device_unique<stripgpu::stripId_t[]>(maxClusters, stream);
  trueCluster_d = cms::cuda::make_device_unique<bool[]>(maxClusters, stream);
  barycenter_d = cms::cuda::make_device_unique<float[]>(maxClusters, stream);

  auto view = cms::cuda::make_host_unique<DeviceView>(stream);
  view->clusterIndex_ = clusterIndex_d.get();
  view->clusterSize_ = clusterSize_d.get();
  view->clusterADCs_ = clusterADCs_d.get();
  view->clusterDetId_ = clusterDetId_d.get();
  view->firstStrip_ = firstStrip_d.get();
  view->trueCluster_ = trueCluster_d.get();
  view->barycenter_ = barycenter_d.get();

  view_d = cms::cuda::make_device_unique<DeviceView>(stream);
  cms::cuda::copyAsync(view_d, view, stream);
}

SiStripClustersCUDA::HostView::HostView(size_t maxClusters, int clustersPerStrip, cudaStream_t stream) {
  clusterIndex_h = cms::cuda::make_host_unique<uint32_t[]>(maxClusters, stream);
  clusterSize_h = cms::cuda::make_host_unique<uint32_t[]>(maxClusters, stream);
  clusterADCs_h = cms::cuda::make_host_unique<uint8_t[]>(maxClusters*clustersPerStrip, stream);
  clusterDetId_h = cms::cuda::make_host_unique<stripgpu::detId_t[]>(maxClusters, stream);
  firstStrip_h = cms::cuda::make_host_unique<stripgpu::stripId_t[]>(maxClusters, stream);
  trueCluster_h = cms::cuda::make_host_unique<bool[]>(maxClusters, stream);
  barycenter_h = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  nClusters_h = maxClusters;
}

std::unique_ptr<SiStripClustersCUDA::HostView> SiStripClustersCUDA::hostView(int clustersPerStrip, cudaStream_t stream) const {
  auto view_h = std::make_unique<HostView>(nClusters_h, clustersPerStrip, stream);

  cms::cuda::copyAsync(view_h->clusterIndex_h, clusterIndex_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->clusterSize_h, clusterSize_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->clusterADCs_h, clusterADCs_d, nClusters_h*clustersPerStrip, stream);
  cms::cuda::copyAsync(view_h->clusterDetId_h, clusterDetId_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->firstStrip_h, firstStrip_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->trueCluster_h, trueCluster_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->barycenter_h, barycenter_d, nClusters_h, stream);

  return view_h;
}