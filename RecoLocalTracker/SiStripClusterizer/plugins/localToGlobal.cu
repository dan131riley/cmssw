#include "CUDADataFormats/SiStripCluster/interface/MkFitSiStripClustersCUDA.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersCUDA.h"

#include "MkFitSiStripHitGPUKernel.h"
#include "localToGlobal.cuh"
#include "clusterGPU.cuh"

//#define GPU_DEBUG
//#define GPU_CHECK

namespace stripgpu {

  //convert detector id to element index in textured arrays
  __host__ __device__ int index_lookup3(const unsigned int detid) {
    int hex4 = (detid >> 16) & 0xF;
    int hex5 = (detid >> 12) & 0xF;
    int hex6 = (detid >> 8) & 0xF;
    int hex7 = (detid >> 4) & 0xF;
    int hex8 = (detid >> 0) & 0xF;

    int shift = -1;
    if (hex4 == 1) {
      if (hex5 == 1) {
        shift = 0;
      } else if (hex5 == 2) {
        shift = 1;
      }
    } else if (hex4 == 0) {
      switch (hex5) {
        case 14:
          shift = 2;
          break;
        case 13:
          shift = 3;
          break;
        case 10:
          shift = 4;
          break;
        case 9:
          shift = 5;
          break;
        case 6:
          shift = 6;
          break;
        case 5:
          shift = 7;
          break;
      }
    }
    return shift * 4096 + hex6 * 256 + hex7 * 16 + hex8;
  }

  __host__ __device__ int index_lookup5(const unsigned int detid) {
    int hex4 = (detid >> 16) & 0xF;
    int hex5 = (detid >> 12) & 0xF;
    int hex6 = (detid >> 8) & 0xF;
    int hex7 = (detid >> 4) & 0xF;
    int hex8 = (detid >> 0) & 0xF;

    int shift = -1;
    if (hex4 == 1) {
      switch (hex5) {
        case 10:
          shift = 0;
          break;
        case 9:
          shift = 1;
          break;
        case 6:
          shift = 2;
          break;
        case 5:
          shift = 3;
          break;
        case 2:
          shift = 4;
          break;
        case 1:
          shift = 5;
          break;
      }
    } else if (hex4 == 0) {
      switch (hex5) {
        case 14:
          shift = 6;
          break;
        case 13:
          shift = 7;
          break;
        case 10:
          shift = 8;
          break;
        case 9:
          shift = 9;
          break;
        case 6:
          shift = 10;
          break;
        case 5:
          shift = 11;
          break;
      }
    }
    return shift * 4096 + hex6 * 256 + hex7 * 16 + hex8;
  }

  __host__ __device__ int index_lookup4(const unsigned int detid) {
    int hex5 = (detid >> 12) & 0xF;
    int hex6 = (detid >> 8) & 0xF;
    int hex7 = (detid >> 4) & 0xF;
    int hex8 = (detid >> 0) & 0xF;

    int shift = -1;
    switch (hex5) {
      case 5:
        shift = 0;
        break;
      case 4:
        shift = 1;
        break;
      case 3:
        shift = 2;
        break;
      case 2:
        shift = 3;
        break;
    }
    return shift * 4096 + hex6 * 256 + hex7 * 16 + hex8;
  }
  __host__ __device__ int index_lookup6(const unsigned int detid) {
    int hex4 = (detid >> 16) & 0xF;
    int hex5 = (detid >> 12) & 0xF;
    int hex6 = (detid >> 8) & 0xF;
    int hex7 = (detid >> 4) & 0xF;
    int hex8 = (detid >> 0) & 0xF;

    int shift = -1;
    switch (hex4) {
      case 10:
        switch (hex5) {
          case 6:
            shift = 0;
            break;
          case 5:
            shift = 1;
            break;
          case 2:
            shift = 2;
            break;
          case 1:
            shift = 3;
            break;
        }
        break;
      case 9:
        switch (hex5) {
          case 14:
            shift = 4;
            break;
          case 13:
            shift = 5;
            break;
          case 10:
            shift = 6;
            break;
          case 9:
            shift = 7;
            break;
          case 6:
            shift = 8;
            break;
          case 5:
            shift = 9;
            break;
          case 2:
            shift = 10;
            break;
          case 1:
            shift = 11;
            break;
        }
        break;
      case 8:
        switch (hex5) {
          case 14:
            shift = 12;
            break;
          case 13:
            shift = 13;
            break;
          case 10:
            shift = 14;
            break;
          case 9:
            shift = 15;
            break;
          case 6:
            shift = 16;
            break;
          case 5:
            shift = 17;
            break;
        }
        break;
      case 6:
        switch (hex5) {
          case 6:
            shift = 18;
            break;
          case 5:
            shift = 19;
            break;
          case 2:
            shift = 20;
            break;
          case 1:
            shift = 21;
            break;
        }
        break;
      case 5:
        switch (hex5) {
          case 14:
            shift = 22;
            break;
          case 13:
            shift = 23;
            break;
          case 10:
            shift = 24;
            break;
          case 9:
            shift = 25;
            break;
          case 6:
            shift = 26;
            break;
          case 5:
            shift = 27;
            break;
          case 2:
            shift = 28;
            break;
          case 1:
            shift = 29;
            break;
        }
        break;
      case 4:
        switch (hex5) {
          case 14:
            shift = 30;
            break;
          case 13:
            shift = 31;
            break;
          case 10:
            shift = 32;
            break;
          case 9:
            shift = 33;
            break;
          case 6:
            shift = 34;
            break;
          case 5:
            shift = 35;
            break;
        }
        break;
    }
    return shift * 4096 + hex6 * 256 + hex7 * 16 + hex8;
  }
  //// return rotations as a double
  //static __inline__ __device__ double fetchRot(texture<int,1> t_h, texture<int,1> t_l, int i){
  //
  //    int hi = tex1Dfetch(t_h,i);
  //    int lo = tex1Dfetch(t_l,i);
  //    return __hiloint2double(hi,lo);
  //
  //}
  //
  //int double2hiint(double val)
  //{
  //    union {
  //        double val;
  //        struct {
  //            int lo;
  //            int hi;
  //        };
  //    } u;
  //    u.val = val;
  //    return u.hi;
  //}
  //
  //int double2loint(double val)
  //{
  //    union {
  //        double val;
  //        struct {
  //            int lo;
  //            int hi;
  //        };
  //    } u;
  //    u.val = val;
  //    return u.lo;
  //}

  __device__ __constant__ float rec_12 = 1.f / 12.f;
  __device__ static void getGlobalBarrel(const unsigned int detid,
                                         const float strip_num,
                                         float* g_x,
                                         float* g_y,
                                         float* g_z,
                                         float* g_xx,
                                         float* g_xy,
                                         float* g_xz,
                                         float* g_yy,
                                         float* g_yz,
                                         float* g_zz,
                                         int elem,
                                         short i,
//                                         float* l_xx,
//                                         float* l_xy,
//                                         float* l_yy,
//                                         float* local,
                                         const LocalToGlobalMap* map_d) {
//    int deti = map_d->det_num_bd[i];
//    if (deti == (int)detid) {
      double pitch = map_d->pitch_bd[i];
      double offset = map_d->offset_bd[i];
      double length = map_d->len_bd[i];
      double pos_x = map_d->pos_x_bd[i];
      double pos_y = map_d->pos_y_bd[i];
      double pos_z = map_d->pos_z_bd[i];
      double drift_x = map_d->drift_x_bd[i];
      double thickness = map_d->thickness_bd[i];
      double backPlane = map_d->backPlane_bd[i];
      double R11 = map_d->R11_bd[i];
      double R12 = map_d->R12_bd[i];
      double R13 = map_d->R13_bd[i];
      double R21 = map_d->R21_bd[i];
      double R22 = map_d->R22_bd[i];
      double R23 = map_d->R23_bd[i];

      double fullProjection = (drift_x * thickness) / pitch;
      double localPoint_corrected = (strip_num - 0.5f * (1.f - backPlane) * fullProjection);
      double localPoint = localPoint_corrected * pitch + offset;
      double localError_xx = pitch * pitch * rec_12;
      double localError_yy = length * length * rec_12;
      double localError_xy = 0;  //rec_12*pitch*length;

      double global_x = R11 * localPoint + pos_x;
      double global_y = R12 * localPoint + pos_y;
      double global_z = R13 * localPoint + pos_z;

      g_x[elem] = R11 * localPoint + pos_x;
      g_y[elem] = R12 * localPoint + pos_y;
      g_z[elem] = R13 * localPoint + pos_z;
      g_xx[elem] =
          R11 * (R11 * localError_xx + R21 * localError_xy) + R21 * (R11 * localError_xy + R21 * localError_yy);
      g_xy[elem] =
          R11 * (R12 * localError_xx + R22 * localError_xy) + R21 * (R12 * localError_xy + R22 * localError_yy);
      g_yy[elem] =
          R12 * (R12 * localError_xx + R22 * localError_xy) + R22 * (R12 * localError_xy + R22 * localError_yy);
      g_xz[elem] =
          R11 * (R13 * localError_xx + R23 * localError_xy) + R21 * (R13 * localError_xy + R23 * localError_yy);
      g_yz[elem] =
          R12 * (R13 * localError_xx + R23 * localError_xy) + R22 * (R13 * localError_xy + R23 * localError_yy);
      g_zz[elem] =
          R13 * (R13 * localError_xx + R23 * localError_xy) + R23 * (R13 * localError_xy + R23 * localError_yy);
//      l_xx[elem] = localError_xx;
//      l_xy[elem] = localError_xy;
//      l_yy[elem] = localError_yy;
//      local[elem] = localPoint;
//    }
  }

  __device__ __constant__ float tanPi8 = 0.4142135623730950;
  __device__ __constant__ float pio8 = 3.141592653589793238 / 8;
  __device__ __constant__ float tan15_val1 = 0.33331906795501708984375;
  __device__ __constant__ float tan15_val2 = 0.135160386562347412109375;
  __device__ __constant__ float atanclip_val1 = 8.05374449538e-2f;
  __device__ __constant__ float atanclip_val2 = 1.38776856032E-1f;
  __device__ __constant__ float atanclip_val3 = 1.99777106478E-1f;
  __device__ __constant__ float atanclip_val4 = 3.33329491539E-1f;
  __device__ static void getGlobalEndcap(const unsigned int detid,
                                         const float strip_num,
                                         float* g_x,
                                         float* g_y,
                                         float* g_z,
                                         float* g_xx,
                                         float* g_xy,
                                         float* g_xz,
                                         float* g_yy,
                                         float* g_yz,
                                         float* g_zz,
                                         int elem,
                                         short i,
//                                         float* l_xx,
//                                         float* l_xy,
//                                         float* l_yy,
//                                         float* local,
                                         const LocalToGlobalMap* map_d) {
//    int deti = map_d->det_num_ed[i];
//    if (deti == (int)detid) {
      int yAx = map_d->yAx_ed[i];
      double rCross = map_d->rCross_ed[i];
      double aw = map_d->aw_ed[i];
      double phi = map_d->phi_ed[i];
      double length = map_d->len_ed[i];
      double pos_x = map_d->pos_x_ed[i];
      double pos_y = map_d->pos_y_ed[i];
      double pos_z = map_d->pos_z_ed[i];
      double R11 = map_d->R11_ed[i];
      double R12 = map_d->R12_ed[i];
      double R13 = map_d->R13_ed[i];
      double R21 = map_d->R21_ed[i];
      double R22 = map_d->R22_ed[i];
      double R23 = map_d->R23_ed[i];
      double drift_x = map_d->drift_x_ed[i];
      double drift_y = map_d->drift_y_ed[i];
      double thickness = map_d->thickness_ed[i];
      double backPlane = map_d->backPlane_ed[i];

      double stripAngle = yAx * (phi + strip_num * aw);
      double tan15 =
          stripAngle * (1 + (stripAngle * stripAngle) * (tan15_val1 + (stripAngle * stripAngle) * tan15_val2));
      double localPoint_uncorrected = yAx * rCross * tan15;
      double x1 = localPoint_uncorrected + 0.5f * drift_x * thickness;
      double x2 = localPoint_uncorrected - 0.5f * drift_x * thickness;
      double y1 = rCross + yAx * 0.5 * drift_y * thickness;
      double fullProjection_t = (y1 * x1 - y1 * x2) / (y1 * y1 + x1 * x2);

      short sgn = fullProjection_t < 0 ? -1 : 1;
      double atanclip = (sgn * fullProjection_t < tanPi8)
                            ? sgn * ((((atanclip_val1 * fullProjection_t * fullProjection_t - atanclip_val2) *
                                           fullProjection_t * fullProjection_t +
                                       atanclip_val3) *
                                          fullProjection_t * fullProjection_t -
                                      atanclip_val4) *
                                         fullProjection_t * fullProjection_t * sgn * fullProjection_t +
                                     sgn * fullProjection_t)
                            : sgn * pio8;
      double fullProjection = atanclip / aw;

      double localPoint_corrected = (strip_num - 0.5f * (1.f - backPlane) * fullProjection);
      double stripAnglex = yAx * (phi + localPoint_corrected * aw);
      double tan15x =
          stripAnglex * (1 + (stripAnglex * stripAnglex) * (tan15_val1 + (stripAnglex * stripAnglex) * tan15_val2));
      double localPoint = yAx * rCross * tan15x;

      double t2 = tan15 * tan15;
      double tt = rec_12 * rCross * rCross * aw * aw;
      double rr = rec_12 * length * length;
      double localError_xx = tt + t2 * rr;
      double localError_yy = tt * t2 + rr;
      double localError_xy = tan15 * (rr - tt);

      g_x[elem] = R11 * localPoint + pos_x;
      g_y[elem] = R12 * localPoint + pos_y;
      g_z[elem] = R13 * localPoint + pos_z;
      g_xx[elem] =
          R11 * (R11 * localError_xx + R21 * localError_xy) + R21 * (R11 * localError_xy + R21 * localError_yy);
      g_xy[elem] =
          R11 * (R12 * localError_xx + R22 * localError_xy) + R21 * (R12 * localError_xy + R22 * localError_yy);
      g_yy[elem] =
          R12 * (R12 * localError_xx + R22 * localError_xy) + R22 * (R12 * localError_xy + R22 * localError_yy);
      g_xz[elem] =
          R11 * (R13 * localError_xx + R23 * localError_xy) + R21 * (R13 * localError_xy + R23 * localError_yy);
      g_yz[elem] =
          R12 * (R13 * localError_xx + R23 * localError_xy) + R22 * (R13 * localError_xy + R23 * localError_yy);
      g_zz[elem] =
          R13 * (R13 * localError_xx + R23 * localError_xy) + R23 * (R13 * localError_xy + R23 * localError_yy);
//      local[elem] = localPoint;
//      l_xx[elem] = localError_xx;
//      l_xy[elem] = localError_xy;
//      l_yy[elem] = localError_yy;
//    }
  }
  __global__ static void localToGlobal(SiStripClustersCUDA::DeviceView* clust_data_d,
                                       MkFitSiStripClustersCUDA::GlobalDeviceView* global_data_d,
                                       const int nStrips,
                                       const LocalToGlobalMap* map_d) {
    float* __restrict__ barycenter = clust_data_d->barycenter_;
    bool* __restrict__ trueCluster = clust_data_d->trueCluster_;
//    float* __restrict__ local_xx = global_data_d->local_xx_;
//    float* __restrict__ local_xy = global_data_d->local_xy_;
//    float* __restrict__ local_yy = global_data_d->local_yy_;
//    float* __restrict__ local = global_data_d->local_;
    float* __restrict__ global_x = global_data_d->global_x_;
    float* __restrict__ global_y = global_data_d->global_y_;
    float* __restrict__ global_z = global_data_d->global_z_;
    float* __restrict__ global_xx = global_data_d->global_xx_;
    float* __restrict__ global_xy = global_data_d->global_xy_;
    float* __restrict__ global_xz = global_data_d->global_xz_;
    float* __restrict__ global_yy = global_data_d->global_yy_;
    float* __restrict__ global_yz = global_data_d->global_yz_;
    float* __restrict__ global_zz = global_data_d->global_zz_;
    short* __restrict__ layer = global_data_d->layer_;
    float* __restrict__ barycenterg = global_data_d->barycenter_;

    auto indexer3 = map_d->indexer3;
    auto indexer4 = map_d->indexer4;
    auto indexer5 = map_d->indexer5;
    auto indexer6 = map_d->indexer6;

    auto detids = clust_data_d->clusterDetId_;
    auto gdetids = global_data_d->clusterDetId_;
   // auto clusterindex = clust_data_d->clusterIndex_;
   // auto gclusterindex = global_data_d->clusterIndex_;
   // auto clusterADCs = clust_data_d->clusterADCs_;
   // auto gclusterADCs = global_data_d->clusterADCs_;
   // auto clusterfirst = clust_data_d->firstStrip_;
   // auto gclusterfirst = global_data_d->firstStrip_;
   // auto clustersize = clust_data_d->clusterSize_;
   // auto gclustersize = global_data_d->clusterSize_;

    static const int kSubDetOffset = 25;
    static const int kSubDetMask = 0x7;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;

    const int i = nthreads * bid + tid;

    //for (int i = 0; i < nStrips; i++) {
    if (i < nStrips) {
      const auto detid = detids[i];
      gdetids[i] = detid;
      layer[i] = -1;
      //if (!trueCluster[i]) {
      //  continue;
      //}
      if (trueCluster[i]) {
      const auto subdet = (detid >> kSubDetOffset) & kSubDetMask;
      short tex_index = -1;
      barycenterg[i] = barycenter[i];
      //gclusterindex[i] = clusterindex[i];
      //gclusterADCs[i] = clusterADCs[i];
      //gclusterfirst[i] = clusterfirst[i];
      //gclustersize[i] = clustersize[i];
      if (subdet == 3) {  //run barrel
        tex_index = indexer3[index_lookup3(detid)];
        getGlobalBarrel(detid,
                        barycenter[i],
                        global_x,
                        global_y,
                        global_z,
                        global_xx,
                        global_xy,
                        global_xz,
                        global_yy,
                        global_yz,
                        global_zz,
                        i,
                        tex_index,
//                        local_xx,
//                        local_xy,
//                        local_yy,
//                        local,
                        map_d);
        layer[i] = ((detid >> 14) & 0x7);
      } else if (subdet == 5) {  // run barrel
        tex_index = indexer5[index_lookup5(detid)];
        getGlobalBarrel(detid,
                        barycenter[i],
                        global_x,
                        global_y,
                        global_z,
                        global_xx,
                        global_xy,
                        global_xz,
                        global_yy,
                        global_yz,
                        global_zz,
                        i,
                        tex_index,
//                        local_xx,
//                        local_xy,
//                        local_yy,
//                        local,
                        map_d);
        layer[i] = ((detid >> 14) & 0x7);
      } else if (subdet == 4) {  //run endcap
        tex_index = indexer4[index_lookup4(detid)];
        getGlobalEndcap(detid,
                        barycenter[i],
                        global_x,
                        global_y,
                        global_z,
                        global_xx,
                        global_xy,
                        global_xz,
                        global_yy,
                        global_yz,
                        global_zz,
                        i,
                        tex_index,
//                        local_xx,
//                        local_xy,
//                        local_yy,
//                        local,
                        map_d);
        layer[i] = ((detid >> 11) & 0x3);
      } else if (subdet == 6) {  // run endcap
        tex_index = indexer6[index_lookup6(detid)];
        getGlobalEndcap(detid,
                        barycenter[i],
                        global_x,
                        global_y,
                        global_z,
                        global_xx,
                        global_xy,
                        global_xz,
                        global_yy,
                        global_yz,
                        global_zz,
                        i,
                        tex_index,
//                        local_xx,
//                        local_xy,
//                        local_yy,
//                        local,
                        map_d);
        layer[i] = ((detid >> 14) & 0xF);
      }
    }
  }
}
  void MkFitSiStripHitGPUKernel::makeGlobal(SiStripClustersCUDA& clusters_d_x,
                                            MkFitSiStripClustersCUDA& clusters_g_x,
                                            cudaStream_t stream) {
    auto clust_data_d = clusters_d_x.view();
    const int nStrips = clusters_d_x.nClusters();
    clusters_g_x = MkFitSiStripClustersCUDA(nStrips, kClusterMaxStrips, stream);
    clusters_g_x.setNClusters(nStrips);
    auto global_data_d = clusters_g_x.gview();
    const int nthreads = 128;
    const int nSeeds = std::min(MAX_SEEDSTRIPS, nStrips);
    const int nblocks = (nStrips + nthreads - 1) / nthreads;
    auto map_d = toDevice();
    localToGlobal<<<nblocks, nthreads, 0, stream>>>(clust_data_d, global_data_d, nStrips, map_d);
    cudaCheck(cudaGetLastError());
  }
}  // namespace stripgpu
