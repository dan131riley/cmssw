#ifndef CUDADataFormats_SiStripCluster_interface_MkFitSiStripClustersCUDA_h
#define CUDADataFormats_SiStripCluster_interface_MkFitSiStripClustersCUDA_h

#include "CUDADataFormats/SiStripCluster/interface/GPUtypes.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include <cuda_runtime.h>

class MkFitSiStripClustersCUDA {
public:
  MkFitSiStripClustersCUDA() = default;
  explicit MkFitSiStripClustersCUDA(size_t maxClusters, int clustersPerStrip, cudaStream_t stream);
  ~MkFitSiStripClustersCUDA() = default;

  MkFitSiStripClustersCUDA(const MkFitSiStripClustersCUDA &) = delete;
  MkFitSiStripClustersCUDA &operator=(const MkFitSiStripClustersCUDA &) = delete;
  MkFitSiStripClustersCUDA(MkFitSiStripClustersCUDA &&) = default;
  MkFitSiStripClustersCUDA &operator=(MkFitSiStripClustersCUDA &&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

  uint32_t nClusters() const { return nClusters_h; }

  class GlobalDeviceView {
  public:
//    __device__ __forceinline__ float local_xx(int i) const { return __ldg(local_xx_ + i); }
//    __device__ __forceinline__ float local_xy(int i) const { return __ldg(local_xy_ + i); }
//    __device__ __forceinline__ float local_yy(int i) const { return __ldg(local_yy_ + i); }
//    __device__ __forceinline__ float local(int i) const { return __ldg(local_ + i); }
    __device__ __forceinline__ float global_x(int i) const { return __ldg(global_x_ + i); }
    __device__ __forceinline__ float global_y(int i) const { return __ldg(global_y_ + i); }
    __device__ __forceinline__ float global_z(int i) const { return __ldg(global_z_ + i); }

    __device__ __forceinline__ float global_xx(int i) const { return __ldg(global_xx_ + i); }
    __device__ __forceinline__ float global_xy(int i) const { return __ldg(global_xy_ + i); }
    __device__ __forceinline__ float global_xz(int i) const { return __ldg(global_xz_ + i); }
    __device__ __forceinline__ float global_yy(int i) const { return __ldg(global_yy_ + i); }
    __device__ __forceinline__ float global_yz(int i) const { return __ldg(global_yz_ + i); }
    __device__ __forceinline__ float global_zz(int i) const { return __ldg(global_zz_ + i); }

    __device__ __forceinline__ short layer(int i) const { return __ldg(layer_ + i); }
    __device__ __forceinline__ float barycenter(int i) const { return __ldg(barycenter_ + i); }  // to remove Tres
//    __device__ __forceinline__ stripgpu::detId_t clusterDetId(int i) const { return __ldg(clusterDetId_ + i); }
//    __device__ __forceinline__ uint32_t clusterIndex(int i) const { return __ldg(clusterIndex_ + i); }
//    __device__ __forceinline__ uint8_t clusterADCs(int i) const { return __ldg(clusterADCs_ + i); }
//    __device__ __forceinline__ stripgpu::stripId_t firstStrip(int i) const { return __ldg(firstStrip_ + i); }
//__device__ __forceinline__ uint32_t clusterSize(int i) const { return __ldg(clusterSize_ + i); }

    friend MkFitSiStripClustersCUDA;

    //   private:
    int nClusters_;

//    float *local_xx_;
//    float *local_xy_;
//    float *local_yy_;
//    float *local_;
    float *global_x_;
    float *global_y_;
    float *global_z_;

    float *global_xx_;
    float *global_xy_;
    float *global_xz_;
    float *global_yy_;
    float *global_yz_;
    float *global_zz_;

    short *layer_;
    float *barycenter_;  // to remove Tres
    stripgpu::detId_t *clusterDetId_;
//    uint32_t *clusterIndex_;
//uint8_t *clusterADCs_;
//stripgpu::stripId_t *firstStrip_;
//uint32_t *clusterSize_;
  };

  GlobalDeviceView *gview() const { return gview_d.get(); }

  class HostView {
  public:
    explicit HostView(size_t maxClusters, int clustersPerStrip, cudaStream_t stream);

    cms::cuda::host::unique_ptr<stripgpu::detId_t[]> clusterDetId_h;
//    cms::cuda::host::unique_ptr<uint32_t[]> clusterIndex_h;
    cms::cuda::host::unique_ptr<float[]> barycenter_h;
//cms::cuda::host::unique_ptr<uint8_t[]> clusterADCs_h;
//cms::cuda::host::unique_ptr<uint32_t[]> clusterSize_h;
//cms::cuda::host::unique_ptr<stripgpu::stripId_t[]> firstStrip_h;

//    cms::cuda::host::unique_ptr<float[]> local_xx_h;
//    cms::cuda::host::unique_ptr<float[]> local_xy_h;
//    cms::cuda::host::unique_ptr<float[]> local_yy_h;
//    cms::cuda::host::unique_ptr<float[]> local_h;
    cms::cuda::host::unique_ptr<float[]> global_x_h;
    cms::cuda::host::unique_ptr<float[]> global_y_h;
    cms::cuda::host::unique_ptr<float[]> global_z_h;
    cms::cuda::host::unique_ptr<float[]> global_xx_h;
    cms::cuda::host::unique_ptr<float[]> global_xy_h;
    cms::cuda::host::unique_ptr<float[]> global_xz_h;
    cms::cuda::host::unique_ptr<float[]> global_yy_h;
    cms::cuda::host::unique_ptr<float[]> global_yz_h;
    cms::cuda::host::unique_ptr<float[]> global_zz_h;

    cms::cuda::host::unique_ptr<short[]> layer_h;
    int nClusters_h;
  };

  std::unique_ptr<HostView> hostView(int clustersPerStrip, cudaStream_t stream) const;

private:
  cms::cuda::device::unique_ptr<stripgpu::detId_t[]> clusterDetId_d;
//  cms::cuda::device::unique_ptr<uint32_t[]> clusterIndex_d;
  cms::cuda::device::unique_ptr<float[]> barycenter_d;
//cms::cuda::device::unique_ptr<uint8_t[]> clusterADCs_d;
//  cms::cuda::device::unique_ptr<uint32_t[]> clusterSize_d;
// cms::cuda::device::unique_ptr<stripgpu::stripId_t[]> firstStrip_d;

//  cms::cuda::device::unique_ptr<float[]> local_xx_d;
//  cms::cuda::device::unique_ptr<float[]> local_xy_d;
//  cms::cuda::device::unique_ptr<float[]> local_yy_d;
//  cms::cuda::device::unique_ptr<float[]> local_d;
  cms::cuda::device::unique_ptr<float[]> global_x_d;
  cms::cuda::device::unique_ptr<float[]> global_y_d;
  cms::cuda::device::unique_ptr<float[]> global_z_d;
  cms::cuda::device::unique_ptr<float[]> global_xx_d;
  cms::cuda::device::unique_ptr<float[]> global_xy_d;
  cms::cuda::device::unique_ptr<float[]> global_xz_d;
  cms::cuda::device::unique_ptr<float[]> global_yy_d;
  cms::cuda::device::unique_ptr<float[]> global_yz_d;
  cms::cuda::device::unique_ptr<float[]> global_zz_d;

  cms::cuda::device::unique_ptr<short[]> layer_d;

  cms::cuda::device::unique_ptr<GlobalDeviceView> gview_d;  // "me" pointer

public:
  int nClusters_h;
};

#endif
