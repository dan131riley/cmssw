#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h

#include "CUDADataFormats/SiStripCluster/interface/GPUtypes.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include <cuda_runtime.h>

class SiStripClustersCUDA {
public:
  SiStripClustersCUDA() = default;
  explicit SiStripClustersCUDA(size_t maxClusters, int clustersPerStrip, cudaStream_t stream);
  ~SiStripClustersCUDA() = default;

  SiStripClustersCUDA(const SiStripClustersCUDA &) = delete;
  SiStripClustersCUDA &operator=(const SiStripClustersCUDA &) = delete;
  SiStripClustersCUDA(SiStripClustersCUDA &&) = default;
  SiStripClustersCUDA &operator=(SiStripClustersCUDA &&) = default;

  void setNClusters(uint32_t nClusters) { nClusters_h = nClusters; }

  uint32_t nClusters() const { return nClusters_h; }

  class DeviceView {
  public:
    __device__ __forceinline__ uint32_t clusterIndex(int i) const { return __ldg(clusterIndex_ + i); }
    __device__ __forceinline__ uint32_t clusterSize(int i) const { return __ldg(clusterSize_ + i); }
    __device__ __forceinline__ uint8_t clusterADCs(int i) const { return __ldg(clusterADCs_ + i); }
    __device__ __forceinline__ stripgpu::detId_t clusterDetId(int i) const { return __ldg(clusterDetId_ + i); }
    __device__ __forceinline__ stripgpu::stripId_t firstStrip(int i) const { return __ldg(firstStrip_ + i); }
    __device__ __forceinline__ bool trueCluster(int i) const { return trueCluster_[i]; }
    __device__ __forceinline__ float barycenter(int i) const  { return __ldg(barycenter_ + i); }

    friend SiStripClustersCUDA;

    //   private:
    uint32_t *clusterIndex_;
    uint32_t *clusterSize_;
    uint8_t *clusterADCs_;
    stripgpu::detId_t *clusterDetId_;
    stripgpu::stripId_t *firstStrip_;
    bool *trueCluster_;
    float *barycenter_;
    int nClusters_;
  };

  DeviceView *view() const { return view_d.get(); }

  class HostView {
  public:
    explicit HostView(size_t maxClusters, int clustersPerStrip, cudaStream_t stream);

    cms::cuda::host::unique_ptr<uint32_t[]> clusterIndex_h;
    cms::cuda::host::unique_ptr<uint32_t[]> clusterSize_h;
    cms::cuda::host::unique_ptr<uint8_t[]> clusterADCs_h;
    cms::cuda::host::unique_ptr<stripgpu::detId_t[]> clusterDetId_h;
    cms::cuda::host::unique_ptr<stripgpu::stripId_t[]> firstStrip_h;
    cms::cuda::host::unique_ptr<bool[]> trueCluster_h;
    cms::cuda::host::unique_ptr<float[]> barycenter_h;
    int nClusters_h;
  };

  std::unique_ptr<HostView> hostView(int clustersPerStrip, cudaStream_t stream) const;

private:
  cms::cuda::device::unique_ptr<uint32_t[]> clusterIndex_d;
  cms::cuda::device::unique_ptr<uint32_t[]> clusterSize_d;
  cms::cuda::device::unique_ptr<uint8_t[]> clusterADCs_d;
  cms::cuda::device::unique_ptr<stripgpu::detId_t[]> clusterDetId_d;
  cms::cuda::device::unique_ptr<stripgpu::stripId_t[]> firstStrip_d;
  cms::cuda::device::unique_ptr<bool[]> trueCluster_d;
  cms::cuda::device::unique_ptr<float[]> barycenter_d;

  cms::cuda::device::unique_ptr<DeviceView> view_d;  // "me" pointer

public:
  int nClusters_h;
};


#endif
