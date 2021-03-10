#include "CUDADataFormats/SiStripCluster/interface/MkFitSiStripClustersCUDA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

MkFitSiStripClustersCUDA::MkFitSiStripClustersCUDA(size_t maxClusters, int clustersPerStrip, cudaStream_t stream) {
  clusterDetId_d = cms::cuda::make_device_unique<stripgpu::detId_t[]>(maxClusters, stream);
  barycenter_d = cms::cuda::make_device_unique<float[]>(maxClusters, stream);

  local_xx_d  = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  local_xy_d  = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  local_yy_d  = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  local_d  = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_x_d  = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_y_d  = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_z_d  = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_xx_d = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_xy_d = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_xz_d = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_yy_d = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_yz_d = cms::cuda::make_device_unique<float[]>(maxClusters, stream);
  global_zz_d = cms::cuda::make_device_unique<float[]>(maxClusters, stream);

  layer_d = cms::cuda::make_device_unique<short[]>(maxClusters, stream);

  auto gview = cms::cuda::make_host_unique<GlobalDeviceView>(stream);
  gview->local_xx_ = local_xx_d.get();
  gview->local_xy_ = local_xy_d.get();
  gview->local_yy_ = local_yy_d.get();
  gview->local_ = local_d.get();
  gview->global_x_ = global_x_d.get();
  gview->global_y_ = global_y_d.get();
  gview->global_z_ = global_z_d.get();
  gview->global_xx_ = global_xx_d.get();
  gview->global_xy_ = global_xy_d.get();
  gview->global_xz_ = global_xz_d.get();
  gview->global_yy_ = global_yy_d.get();
  gview->global_yz_ = global_yz_d.get();
  gview->global_zz_ = global_zz_d.get();
  gview->barycenter_ = barycenter_d.get();//to remove Tres
  gview->clusterDetId_ = clusterDetId_d.get();

  gview->layer_ = layer_d.get();

  gview_d = cms::cuda::make_device_unique<GlobalDeviceView>(stream);
  cms::cuda::copyAsync(gview_d, gview, stream);
}

MkFitSiStripClustersCUDA::HostView::HostView(size_t maxClusters, int clustersPerStrip, cudaStream_t stream) {
  clusterDetId_h = cms::cuda::make_host_unique<stripgpu::detId_t[]>(maxClusters, stream);
  barycenter_h = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  
  local_xx_h  = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  local_xy_h  = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  local_yy_h  = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  local_h  = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_x_h  = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_y_h  = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_z_h  = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_xx_h = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_xy_h = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_xz_h = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_yy_h = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_yz_h = cms::cuda::make_host_unique<float[]>(maxClusters, stream);
  global_zz_h = cms::cuda::make_host_unique<float[]>(maxClusters, stream);

  layer_h = cms::cuda::make_host_unique<short[]>(maxClusters, stream);

  nClusters_h = maxClusters;
}

std::unique_ptr<MkFitSiStripClustersCUDA::HostView> MkFitSiStripClustersCUDA::hostView(int clustersPerStrip, cudaStream_t stream) const {
  auto view_h = std::make_unique<HostView>(nClusters_h, clustersPerStrip, stream);

  cms::cuda::copyAsync(view_h->clusterDetId_h, clusterDetId_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->barycenter_h, barycenter_d, nClusters_h, stream);
  
  cms::cuda::copyAsync(view_h->local_xx_h, local_xx_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->local_xy_h, local_xy_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->local_yy_h, local_yy_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->local_h, local_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_x_h, global_x_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_y_h, global_y_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_z_h, global_z_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_xx_h, global_xx_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_xy_h, global_xy_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_xz_h, global_xz_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_yy_h, global_yy_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_yz_h, global_yz_d, nClusters_h, stream);
  cms::cuda::copyAsync(view_h->global_zz_h, global_zz_d, nClusters_h, stream);

  cms::cuda::copyAsync(view_h->layer_h, layer_d, nClusters_h, stream);

  return view_h;
}
