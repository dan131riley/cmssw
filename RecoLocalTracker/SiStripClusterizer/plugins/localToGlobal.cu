#include <stdio.h>
#include <cub/cub.cuh>

#include "HeterogeneousCore/CUDAUtilities/interface/allocate_device.h"
#include "HeterogeneousCore/CUDAUtilities/interface/allocate_host.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

#include "ChanLocsGPU.h"
#include "SiStripRawToClusterGPUKernel.h"
#include "clusterGPU.cuh"
#include "localToGlobal.cuh"

#include "MkFitSiStripHitGPUKernel.h"
#include "Geometry/CommonTopologies/interface/BowedSurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

//#define GPU_DEBUG
//#define GPU_CHECK

namespace stripgpu {
  __device__ constexpr int maxseeds() { return MAX_SEEDSTRIPS; }


//convert detector id to element index in textured arrays
__host__ __device__ int index_lookup3(const unsigned int detid){

  int hex4 = (detid >> 16) & 0xF;
  int hex5 = (detid >> 12) & 0xF;
  int hex6 = (detid >> 8) & 0xF;
  int hex7 = (detid >> 4) & 0xF;
  int hex8 = (detid >> 0) & 0xF;

  int shift = -1;
    if (hex4 == 1){
      if(hex5 == 1){ shift = 0;}
      else if (hex5 ==2){ shift = 1;}
    }else if (hex4 == 0){
      switch(hex5){
        case 14: shift = 2;break;
        case 13: shift = 3;break;
        case 10: shift = 4;break;
        case 9: shift = 5;break;
        case 6: shift = 6;break;
        case 5: shift = 7;break;
      }
    }
    return shift*4096 + hex6*256 + hex7*16 + hex8;
}

__host__ __device__ int index_lookup5(const unsigned int detid){

  int hex4 = (detid >> 16) & 0xF;
  int hex5 = (detid >> 12) & 0xF;
  int hex6 = (detid >> 8) & 0xF;
  int hex7 = (detid >> 4) & 0xF;
  int hex8 = (detid >> 0) & 0xF;

  int shift = -1;
    if (hex4 == 1){
      switch(hex5){
        case 10: shift = 0;break;
        case 9: shift = 1;break;
        case 6: shift = 2;break;
        case 5: shift = 3;break;
        case 2: shift = 4;break;
        case 1: shift = 5;break;
      }
    }else if(hex4 == 0){
      switch(hex5){
        case 14: shift = 6;break;
        case 13: shift = 7;break;
        case 10: shift = 8;break;
        case 9: shift = 9;break;
        case 6: shift = 10;break;
        case 5: shift = 11;break;
      }
    }
    return shift*4096 + hex6*256 + hex7*16 + hex8;
}

__host__ __device__ int index_lookup4(const unsigned int detid){

  int hex5 = (detid >> 12) & 0xF;
  int hex6 = (detid >> 8) & 0xF;
  int hex7 = (detid >> 4) & 0xF;
  int hex8 = (detid >> 0) & 0xF;

  int shift = -1;
      switch(hex5){
        case 5: shift = 0;break;
        case 4: shift = 1;break;
        case 3: shift = 2;break;
        case 2: shift = 3;break;
      }
    return shift*4096 + hex6*256 + hex7*16 + hex8;
}
__host__ __device__ int index_lookup6(const unsigned int detid){

  int hex4 = (detid >> 16) & 0xF;
  int hex5 = (detid >> 12) & 0xF;
  int hex6 = (detid >> 8) & 0xF;
  int hex7 = (detid >> 4) & 0xF;
  int hex8 = (detid >> 0) & 0xF;

  int shift = -1;
    switch(hex4){
      case 10:
      switch(hex5){
        case 6: shift = 0;break;
        case 5: shift = 1;break;
        case 2: shift = 2;break;
        case 1: shift = 3;break;
      }
      break;
     case 9:
      switch(hex5){
        case 14: shift = 4;break;
        case 13: shift = 5;break;
        case 10: shift = 6;break;
        case 9: shift = 7;break;
        case 6: shift = 8;break;
        case 5: shift = 9;break;
        case 2: shift = 10;break;
        case 1: shift = 11;break;
      }
      break;
     case 8:
      switch(hex5){
        case 14: shift = 12;break;
        case 13: shift = 13;break;
        case 10: shift = 14;break;
        case 9: shift = 15;break;
        case 6: shift = 16;break;
        case 5: shift = 17;break;
      }
      break;
     case 6:
      switch(hex5){
        case 6: shift = 18;break;
        case 5: shift = 19;break;
        case 2: shift = 20;break;
        case 1: shift = 21;break;
      }
      break;
     case 5 :
      switch(hex5){
        case 14: shift = 22;break;
        case 13: shift = 23;break;
        case 10: shift = 24;break;
        case 9: shift = 25;break;
        case 6: shift = 26;break;
        case 5: shift = 27;break;
        case 2: shift = 28;break;
        case 1: shift = 29;break;
      }
      break;
     case 4 :
      switch(hex5){
        case 14: shift = 30;break;
        case 13: shift = 31;break;
        case 10: shift = 32;break;
        case 9: shift = 33;break;
        case 6: shift = 34;break;
        case 5: shift = 35;break;
      }
      break;
    }
    return shift*4096 + hex6*256 + hex7*16 + hex8;
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

__device__ __constant__ float rec_12 = 1.f/12.f;
__device__ static void getGlobalBarrel(const unsigned int detid, const float strip_num
                                       ,float* g_x, float* g_y, float* g_z,
                                       float* g_xx, float* g_xy, float* g_xz,
                                       float* g_yy, float* g_yz, float* g_zz,int elem, short i
                                       ,float* l_xx, float* l_xy, float* l_yy, float* local
){
      int deti = det_num_bd[i];
      if (deti == (int)detid){
        double pitch = pitch_bd[i];
        double offset = offset_bd[i];
        double length = len_bd[i];
        double pos_x = pos_x_bd[i];
        double pos_y = pos_y_bd[i];
        double pos_z = pos_z_bd[i];
        double drift_x = drift_x_bd[i];
        double thickness = thickness_bd[i];
        double backPlane = backPlane_bd[i];
        double R11 = R11_bd[i];
        double R12 = R12_bd[i];
        double R13 = R13_bd[i];
        double R21 = R21_bd[i];
        double R22 = R22_bd[i];
        double R23 = R23_bd[i];

        double fullProjection = (drift_x*thickness)/pitch;
        double localPoint_corrected = (strip_num - 0.5f*(1.f - backPlane)*fullProjection);
        double localPoint = localPoint_corrected*pitch+offset;
        double localError_xx = pitch*pitch*rec_12;
        double localError_yy = length*length*rec_12;
        double localError_xy = 0;//rec_12*pitch*length;

        double global_x = R11 * localPoint + pos_x;
        double global_y = R12 * localPoint + pos_y;
        double global_z = R13 * localPoint + pos_z;

        g_x[elem] = R11 * localPoint + pos_x;
        g_y[elem] = R12 * localPoint + pos_y;
        g_z[elem] = R13 * localPoint + pos_z;
        g_xx[elem] = R11*(R11*localError_xx + R21*localError_xy) + R21*( R11*localError_xy + R21*localError_yy);
        g_xy[elem] = R11*(R12*localError_xx + R22*localError_xy) + R21*( R12*localError_xy + R22*localError_yy);
        g_yy[elem] = R12*(R12*localError_xx + R22*localError_xy) + R22*( R12*localError_xy + R22*localError_yy);
        g_xz[elem] = R11*(R13*localError_xx + R23*localError_xy) + R21*( R13*localError_xy + R23*localError_yy);
        g_yz[elem] = R12*(R13*localError_xx + R23*localError_xy) + R22*( R13*localError_xy + R23*localError_yy);
        g_zz[elem] = R13*(R13*localError_xx + R23*localError_xy) + R23*( R13*localError_xy + R23*localError_yy);
        l_xx[elem] = localError_xx;
        l_xy[elem] = localError_xy;
        l_yy[elem] = localError_yy;
        local[elem] = localPoint;
    }
}

__device__ __constant__ float tanPi8 = 0.4142135623730950;
__device__ __constant__ float pio8 = 3.141592653589793238 / 8;
__device__ __constant__ float tan15_val1 = 0.33331906795501708984375; 
__device__ __constant__ float tan15_val2 = 0.135160386562347412109375; 
__device__ __constant__ float atanclip_val1 = 8.05374449538e-2f; 
__device__ __constant__ float atanclip_val2 = 1.38776856032E-1f; 
__device__ __constant__ float atanclip_val3 = 1.99777106478E-1f;
__device__ __constant__ float atanclip_val4 = 3.33329491539E-1f;
__device__ static void getGlobalEndcap(const unsigned int detid, const float strip_num
                                       ,float* g_x, float* g_y, float* g_z,
                                       float* g_xx, float* g_xy, float* g_xz,
                                       float* g_yy, float* g_yz, float* g_zz,int elem, short i
                                       ,float* l_xx, float* l_xy, float* l_yy, float* local
){
      int deti = det_num_ed[i];
      if (deti == (int)detid){
        int yAx = yAx_ed[i];
        double rCross = rCross_ed[i];
        double aw = aw_ed[i];
        double phi = phi_ed[i];
        double length =len_ed[i];
        double pos_x = pos_x_ed[i];
        double pos_y = pos_y_ed[i];
        double pos_z = pos_z_ed[i];
        double R11 = R11_ed[i];
        double R12 = R12_ed[i];
        double R13 = R13_ed[i];
        double R21 = R21_ed[i];
        double R22 = R22_ed[i];
        double R23 = R23_ed[i];
        double drift_x = drift_x_ed[i];
        double drift_y = drift_y_ed[i];
        double thickness = thickness_ed[i];
        double backPlane = backPlane_ed[i];

        double stripAngle = yAx *(phi+strip_num*aw);
        double tan15 = stripAngle *(1 + (stripAngle*stripAngle) * (tan15_val1 + (stripAngle * stripAngle) * tan15_val2));
        double localPoint_uncorrected = yAx * rCross* tan15;
        double x1 =localPoint_uncorrected + 0.5f*drift_x*thickness;
        double x2 =localPoint_uncorrected - 0.5f*drift_x*thickness;
        double y1 =rCross+yAx*0.5*drift_y*thickness;
        double fullProjection_t = (y1 * x1 - y1 * x2) / (y1 * y1 + x1 * x2);

        short sgn = fullProjection_t<0 ? -1: 1;
        double atanclip = (sgn*fullProjection_t < tanPi8) ? sgn*((((atanclip_val1 * fullProjection_t*fullProjection_t - atanclip_val2) * fullProjection_t*fullProjection_t + atanclip_val3) * fullProjection_t*fullProjection_t - atanclip_val4) * fullProjection_t * fullProjection_t* sgn*fullProjection_t + sgn*fullProjection_t) : sgn*pio8;
        double fullProjection = atanclip /aw;

        double localPoint_corrected = (strip_num - 0.5f*(1.f - backPlane)*fullProjection);
        double stripAnglex = yAx *(phi+localPoint_corrected*aw);
        double tan15x = stripAnglex *(1 + (stripAnglex*stripAnglex) * (tan15_val1 + (stripAnglex * stripAnglex) * tan15_val2));
        double localPoint = yAx * rCross* tan15x;

        double t2 = tan15*tan15;
        double tt = rec_12 * rCross*rCross*aw*aw;
        double rr = rec_12*length*length;
        double localError_xx = tt+ t2*rr;
        double localError_yy = tt*t2+rr;
        double localError_xy = tan15*(rr-tt);

        g_x[elem] = R11 * localPoint + pos_x;
        g_y[elem] = R12 * localPoint + pos_y;
        g_z[elem] = R13 * localPoint + pos_z;
        g_xx[elem] = R11*(R11*localError_xx + R21*localError_xy) + R21*( R11*localError_xy + R21*localError_yy);
        g_xy[elem] = R11*(R12*localError_xx + R22*localError_xy) + R21*( R12*localError_xy + R22*localError_yy);
        g_yy[elem] = R12*(R12*localError_xx + R22*localError_xy) + R22*( R12*localError_xy + R22*localError_yy);
        g_xz[elem] = R11*(R13*localError_xx + R23*localError_xy) + R21*( R13*localError_xy + R23*localError_yy);
        g_yz[elem] = R12*(R13*localError_xx + R23*localError_xy) + R22*( R13*localError_xy + R23*localError_yy);
        g_zz[elem] = R13*(R13*localError_xx + R23*localError_xy) + R23*( R13*localError_xy + R23*localError_yy);
        local[elem] = localPoint;
        l_xx[elem] = localError_xx;
        l_xy[elem] = localError_xy;
        l_yy[elem] = localError_yy;
    }
}
  __global__
  static void localToGlobal( SiStripClustersCUDA::DeviceView *clust_data_d
                            ,MkFitSiStripClustersCUDA::GlobalDeviceView *global_data_d
                            ,const int nStrips
){
    float *__restrict__ barycenter = clust_data_d->barycenter_;
    bool *__restrict__ trueCluster = clust_data_d->trueCluster_;
    float *__restrict__ local_xx =  global_data_d->local_xx_;
    float *__restrict__ local_xy =  global_data_d->local_xy_;
    float *__restrict__ local_yy =  global_data_d->local_yy_;
    float *__restrict__ local =  global_data_d->local_;
    float *__restrict__ global_x =  global_data_d->global_x_;
    float *__restrict__ global_y =  global_data_d->global_y_;
    float *__restrict__ global_z =  global_data_d->global_z_;
    float *__restrict__ global_xx = global_data_d->global_xx_;
    float *__restrict__ global_xy = global_data_d->global_xy_;
    float *__restrict__ global_xz = global_data_d->global_xz_;
    float *__restrict__ global_yy = global_data_d->global_yy_;
    float *__restrict__ global_yz = global_data_d->global_yz_;
    float *__restrict__ global_zz = global_data_d->global_zz_;
    short *__restrict__ layer = global_data_d->layer_;
    float *__restrict__ barycenterg = global_data_d->barycenter_;

    auto detids = clust_data_d->clusterDetId_;
    auto gdetids = global_data_d->clusterDetId_;

    static const int kSubDetOffset = 25;
    static const int kSubDetMask = 0x7;
    for (int i =0; i<nStrips; i++){
          const auto detid = detids[i];
          gdetids[i] = detid;
          const auto subdet = (detid >> kSubDetOffset) & kSubDetMask;
          short tex_index = -1;
          layer[i] = -1;
          if(!trueCluster[i]){continue;}
          barycenterg[i] = barycenter[i];
          if (subdet == 3) { //run barrel
            tex_index = indexer3[index_lookup3(detid)];
            getGlobalBarrel(detid,barycenter[i]
                            ,global_x,global_y,global_z,
                            global_xx,global_xy,global_xz,
                            global_yy,global_yz,global_zz,i,tex_index
                            ,local_xx,local_xy,local_yy,local
                            );
            layer[i] = ((detid >>14) & 0x7);
          } else if( subdet == 5 ){// run barrel
            tex_index = indexer5[index_lookup5(detid)];
            getGlobalBarrel(detid,barycenter[i]
                            ,global_x,global_y,global_z,
                            global_xx,global_xy,global_xz,
                            global_yy,global_yz,global_zz,i,tex_index
                            ,local_xx,local_xy,local_yy,local
                            );
            layer[i] = ((detid >>14) & 0x7);
          } else if (subdet == 4) { //run endcap
            tex_index = indexer4[index_lookup4(detid)];
            getGlobalEndcap(detid,barycenter[i]
                            ,global_x,global_y,global_z,
                            global_xx,global_xy,global_xz,
                            global_yy,global_yz,global_zz,i,tex_index
                            ,local_xx,local_xy,local_yy,local
                            );
            layer[i] = ((detid >>11) & 0x3);
          } else if(subdet ==6 ){// run endcap
            tex_index = indexer6[index_lookup6(detid)];
            getGlobalEndcap(detid,barycenter[i]
                            ,global_x,global_y,global_z,
                            global_xx,global_xy,global_xz,
                            global_yy,global_yz,global_zz,i,tex_index
                            ,local_xx,local_xy,local_yy,local
                            );
            layer[i] = ((detid >>14) & 0xF);
          }
    }
}

  void MkFitSiStripHitGPUKernel::loadBarrel(const std::vector<const GeometricDet*> dets_barrel, /*const std::vector<const GeomDet*> rots_barrel,*/ const SiStripBackPlaneCorrection* BackPlaneCorrectionMap, const MagneticField* MagFieldMap,const SiStripLorentzAngle* LorentzAngleMap, const std::vector<std::tuple<unsigned int, float, float , float, float, float , float, float, float , float, float, float , float, float, float , float>> stripUnit){
    printf("LOADING BARREL\n");
    short* indexer3_h;
    short* indexer5_h;
    cudaMallocHost((void**)&indexer3_h,31214*sizeof(short));
    cudaMallocHost((void**)&indexer5_h,46426*sizeof(short));
    cudaMemset(indexer3_h,-1,31214*sizeof(short));
    cudaMemset(indexer5_h,-1,46426*sizeof(short));

    int* det_num_h;
    double* pitch_h;
    double* offset_h;
    double* len_h;
    double* pos_x_h;
    double* pos_y_h;
    double* pos_z_h;
    double* R11_h;
    double* R12_h;
    double* R13_h;
    double* R21_h;
    double* R22_h;
    double* R23_h;
    double* backPlane_h;
    double* thickness_h;
    double* drift_x_h;
    cudaMallocHost((void**)&det_num_h,DETS_barrel*sizeof(int));
    cudaMallocHost((void**)&pitch_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&offset_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&len_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&pos_x_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&pos_y_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&pos_z_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&R11_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&R12_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&R13_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&R21_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&R22_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&R23_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&backPlane_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&drift_x_h,DETS_barrel*sizeof(double));
    cudaMallocHost((void**)&thickness_h,DETS_barrel*sizeof(double));

    for (auto it = dets_barrel.begin(); it != dets_barrel.end();++it){
      int i = std::distance(dets_barrel.begin(),it);
      const GeometricDet* det = dets_barrel[i];
      det_num_h[i] = det->geographicalID().rawId();

      int nstrip = int(128* det->siliconAPVNum());
      std::unique_ptr<const Bounds> bounds(det->bounds());
      len_h[i] = (&(*bounds))->length();
      double width = (&(*bounds))->width();
      thickness_h[i] = (&(*bounds))->thickness();
      pitch_h[i] = width/nstrip;
      offset_h[i] = -0.5*width;
      int sub = (det_num_h[i] >> 25) & 0x7;
      if(sub == 3){
        indexer3_h[index_lookup3(det_num_h[i])] = i;
      }else if(sub == 5){
        indexer5_h[index_lookup5(det_num_h[i])] = i;
      }
    }

    for (auto it = stripUnit.begin(); it != stripUnit.end();++it){
      int j = std::distance(stripUnit.begin(),it);
      const auto dus = stripUnit[j];
      
      auto rot_num = std::get<0>(dus);
      int i = -1;
      int sub = (rot_num >> 25) & 0x7;
      if(sub == 3){
        int lookup = index_lookup3(rot_num);
        if(lookup > 31214) {continue;}
        i = indexer3_h[lookup];
      }
      if(sub == 5){
        int lookup = index_lookup5(rot_num);
        if(lookup > 46426) {continue;}
        i = indexer5_h[lookup];
      }
      if (i == -1){ continue;}
      backPlane_h[i] = BackPlaneCorrectionMap->getBackPlaneCorrection(rot_num);
      double lorentzAngle = LorentzAngleMap->getLorentzAngle(rot_num);
      drift_x_h[i] = -lorentzAngle*std::get<2>(dus);
      pos_x_h[i] = std::get<4>(dus);
      pos_y_h[i] = std::get<5>(dus);
      pos_z_h[i] = std::get<6>(dus);
      R11_h[i] = std::get<7>(dus);
      R12_h[i] = std::get<8>(dus);
      R13_h[i] = std::get<9>(dus);
      R21_h[i] = std::get<10>(dus);
      R22_h[i] = std::get<11>(dus);
      R23_h[i] = std::get<12>(dus);
    }
    short* indexer3_hd;
    cudaGetSymbolAddress((void**)&indexer3_hd,indexer3);
    cudaMemcpy(indexer3_hd,indexer3_h,31214*sizeof(short),cudaMemcpyHostToDevice);
    short* indexer5_hd;
    cudaGetSymbolAddress((void**)&indexer5_hd,indexer5);
    cudaMemcpy(indexer5_hd,indexer5_h,46426*sizeof(short),cudaMemcpyHostToDevice);

    int* det_num_hd;
    cudaGetSymbolAddress((void**)&det_num_hd,det_num_bd);
    cudaMemcpy(det_num_hd,det_num_h,DETS_barrel*sizeof(int),cudaMemcpyHostToDevice);
    double* pitch_hd;
    cudaGetSymbolAddress((void**)&pitch_hd,pitch_bd);
    cudaMemcpy(pitch_hd,pitch_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* offset_hd;
    cudaGetSymbolAddress((void**)&offset_hd,offset_bd);
    cudaMemcpy(offset_hd,offset_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* len_hd;
    cudaGetSymbolAddress((void**)&len_hd,len_bd);
    cudaMemcpy(len_hd,len_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* pos_x_hd;
    cudaGetSymbolAddress((void**)&pos_x_hd,pos_x_bd);
    cudaMemcpy(pos_x_hd,pos_x_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* pos_y_hd;
    cudaGetSymbolAddress((void**)&pos_y_hd,pos_y_bd);
    cudaMemcpy(pos_y_hd,pos_y_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* pos_z_hd;
    cudaGetSymbolAddress((void**)&pos_z_hd,pos_z_bd);
    cudaMemcpy(pos_z_hd,pos_z_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* R11_hd;
    cudaGetSymbolAddress((void**)&R11_hd,R11_bd);
    cudaMemcpy(R11_hd,R11_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* R12_hd;
    cudaGetSymbolAddress((void**)&R12_hd,R12_bd);
    cudaMemcpy(R12_hd,R12_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* R13_hd;
    cudaGetSymbolAddress((void**)&R13_hd,R13_bd);
    cudaMemcpy(R13_hd,R13_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* R21_hd;
    cudaGetSymbolAddress((void**)&R21_hd,R21_bd);
    cudaMemcpy(R21_hd,R21_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* R22_hd;
    cudaGetSymbolAddress((void**)&R22_hd,R22_bd);
    cudaMemcpy(R22_hd,R22_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* R23_hd;
    cudaGetSymbolAddress((void**)&R23_hd,R23_bd);
    cudaMemcpy(R23_hd,R23_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* backPlane_hd;
    cudaGetSymbolAddress((void**)&backPlane_hd,backPlane_bd);
    cudaMemcpy(backPlane_hd,backPlane_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* thickness_hd;
    cudaGetSymbolAddress((void**)&thickness_hd,thickness_bd);
    cudaMemcpy(thickness_hd,thickness_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);
    double* drift_x_hd;
    cudaGetSymbolAddress((void**)&drift_x_hd,drift_x_bd);
    cudaMemcpy(drift_x_hd,drift_x_h,DETS_barrel*sizeof(double),cudaMemcpyHostToDevice);

}
  void MkFitSiStripHitGPUKernel::loadEndcap(const std::vector<const GeometricDet*> dets_endcap, /*const std::vector<const GeomDet*> rots_endcap,*/ const SiStripBackPlaneCorrection* BackPlaneCorrectionMap, const MagneticField* MagFieldMap,const SiStripLorentzAngle* LorentzAngleMap, const std::vector<std::tuple<unsigned int, float, float , float, float, float , float, float, float , float, float, float , float, float, float , float>> stripUnit){
    printf("LOADING EndCap\n");
    short* indexer4_h;
    short* indexer6_h;
    cudaMallocHost((void**)&indexer4_h,16208*sizeof(short));
    cudaMallocHost((void**)&indexer6_h,145652*sizeof(short));
    cudaMemset(indexer4_h,-1,16208*sizeof(short));
    cudaMemset(indexer6_h,-1,145652*sizeof(short));

    int* det_num_h;
    int* yAx_h;
    double* backPlane_h;
    double* thickness_h;
    double* drift_x_h;
    double* drift_y_h;
    double* rCross_h;
    double* aw_h;
    double* phi_h;
    double* len_h;
    double* pos_x_h;
    double* pos_y_h;
    double* pos_z_h;
    double* R11_h;
    double* R12_h;
    double* R13_h;
    double* R21_h;
    double* R22_h;
    double* R23_h;
    cudaMallocHost((void**)&det_num_h,DETS_endcap*sizeof(int));
    cudaMallocHost((void**)&yAx_h,DETS_endcap*sizeof(int));
    cudaMallocHost((void**)&rCross_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&aw_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&phi_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&len_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&pos_x_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&pos_y_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&pos_z_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&R11_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&R12_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&R13_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&R21_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&R22_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&R23_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&backPlane_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&drift_x_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&drift_y_h,DETS_endcap*sizeof(double));
    cudaMallocHost((void**)&thickness_h,DETS_endcap*sizeof(double));

    for (auto it = dets_endcap.begin(); it != dets_endcap.end();++it){
      int i = std::distance(dets_endcap.begin(),it);
      const GeometricDet* det = dets_endcap[i];
      det_num_h[i] = det->geographicalId().rawId();

      int nstrip = int(128 *det->siliconAPVNum());
      std::unique_ptr<const Bounds> bounds(det->bounds());
      yAx_h[i] = (dynamic_cast<const TrapezoidalPlaneBounds*>(&(*bounds)))->yAxisOrientation();
      len_h[i] = (&(*bounds))->length();
      thickness_h[i] = (&(*bounds))->thickness();
      float width = (&(*bounds))->width();
      float w_halfl = (&(*bounds))->widthAtHalfLength();
      rCross_h[i] = w_halfl * len_h[i]/(2*(width-w_halfl));
      aw_h[i] = atan2(w_halfl/2., static_cast<float>(rCross_h[i]))/(nstrip/2);
      phi_h[i] = -(0.5 *nstrip) *aw_h[i];

      int sub = (det_num_h[i] >> 25) & 0x7;
      if(sub == 4){
        indexer4_h[index_lookup4(det_num_h[i])] = i;
      }else if(sub == 6){
        indexer6_h[index_lookup6(det_num_h[i])] = i;
      }
    }
    for (auto it = stripUnit.begin(); it != stripUnit.end();++it){
      int j = std::distance(stripUnit.begin(),it);
      const auto dus = stripUnit[j];
      
      auto rot_num = std::get<0>(dus);
      int i = -1;
      int sub = (rot_num >> 25) & 0x7;
      if(sub == 4){
        int lookup = index_lookup4(rot_num);
        if (lookup > 16208){continue;}
        i = indexer4_h[lookup];
      }
      if(sub == 6){
        int lookup = index_lookup6(rot_num);
        if (lookup > 145652){continue;}
        i = indexer6_h[lookup];
      }
      if (i == -1){ continue;}
      backPlane_h[i] = BackPlaneCorrectionMap->getBackPlaneCorrection(rot_num);
      double lorentzAngle = LorentzAngleMap->getLorentzAngle(rot_num);
      drift_x_h[i] = -lorentzAngle*std::get<2>(dus);
      drift_y_h[i] = lorentzAngle*std::get<1>(dus);
      pos_x_h[i] = std::get<4>(dus);
      pos_y_h[i] = std::get<5>(dus);
      pos_z_h[i] = std::get<6>(dus);
      R11_h[i] = std::get<7>(dus);
      R12_h[i] = std::get<8>(dus);
      R13_h[i] = std::get<9>(dus);
      R21_h[i] = std::get<10>(dus);
      R22_h[i] = std::get<11>(dus);
      R23_h[i] = std::get<12>(dus);
    }
    short* indexer4_hd;
    cudaGetSymbolAddress((void**)&indexer4_hd,indexer4);
    cudaMemcpy(indexer4_hd,indexer4_h,16208*sizeof(short),cudaMemcpyHostToDevice);
    short* indexer6_hd;
    cudaGetSymbolAddress((void**)&indexer6_hd,indexer6);
    cudaMemcpy(indexer6_hd,indexer6_h,145652*sizeof(short),cudaMemcpyHostToDevice);

    int* det_num_hd;
    cudaGetSymbolAddress((void**)&det_num_hd,det_num_ed);
    cudaMemcpy(det_num_hd,det_num_h,DETS_endcap*sizeof(int),cudaMemcpyHostToDevice);
    int* yAx_hd;
    cudaGetSymbolAddress((void**)&yAx_hd,yAx_ed);
    cudaMemcpy(yAx_hd,yAx_h,DETS_endcap*sizeof(int),cudaMemcpyHostToDevice);
    double* rCross_hd;
    cudaGetSymbolAddress((void**)&rCross_hd,rCross_ed);
    cudaMemcpy(rCross_hd,rCross_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* aw_hd;
    cudaGetSymbolAddress((void**)&aw_hd,aw_ed);
    cudaMemcpy(aw_hd,aw_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* phi_hd;
    cudaGetSymbolAddress((void**)&phi_hd,phi_ed);
    cudaMemcpy(phi_hd,phi_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* len_hd;
    cudaGetSymbolAddress((void**)&len_hd,len_ed);
    cudaMemcpy(len_hd,len_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);

    double* pos_x_hd;
    cudaGetSymbolAddress((void**)&pos_x_hd,pos_x_ed);
    cudaMemcpy(pos_x_hd,pos_x_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* pos_y_hd;
    cudaGetSymbolAddress((void**)&pos_y_hd,pos_y_ed);
    cudaMemcpy(pos_y_hd,pos_y_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* pos_z_hd;
    cudaGetSymbolAddress((void**)&pos_z_hd,pos_z_ed);
    cudaMemcpy(pos_z_hd,pos_z_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* R11_hd;
    cudaGetSymbolAddress((void**)&R11_hd,R11_ed);
    cudaMemcpy(R11_hd,R11_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* R12_hd;
    cudaGetSymbolAddress((void**)&R12_hd,R12_ed);
    cudaMemcpy(R12_hd,R12_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* R13_hd;
    cudaGetSymbolAddress((void**)&R13_hd,R13_ed);
    cudaMemcpy(R13_hd,R13_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* R21_hd;
    cudaGetSymbolAddress((void**)&R21_hd,R21_ed);
    cudaMemcpy(R21_hd,R21_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* R22_hd;
    cudaGetSymbolAddress((void**)&R22_hd,R22_ed);
    cudaMemcpy(R22_hd,R22_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* R23_hd;
    cudaGetSymbolAddress((void**)&R23_hd,R23_ed);
    cudaMemcpy(R23_hd,R23_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* backPlane_hd;
    cudaGetSymbolAddress((void**)&backPlane_hd,backPlane_ed);
    cudaMemcpy(backPlane_hd,backPlane_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* thickness_hd;
    cudaGetSymbolAddress((void**)&thickness_hd,thickness_ed);
    cudaMemcpy(thickness_hd,thickness_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* drift_x_hd;
    cudaGetSymbolAddress((void**)&drift_x_hd,drift_x_ed);
    cudaMemcpy(drift_x_hd,drift_x_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
    double* drift_y_hd;
    cudaGetSymbolAddress((void**)&drift_y_hd,drift_y_ed);
    cudaMemcpy(drift_y_hd,drift_y_h,DETS_endcap*sizeof(double),cudaMemcpyHostToDevice);
  }

   
  void MkFitSiStripHitGPUKernel::makeGlobal(SiStripClustersCUDA& clusters_d_x,MkFitSiStripClustersCUDA& clusters_g_x, cudaStream_t stream) {
    
    auto clust_data_d = clusters_d_x.view();
    const int nStrips = clusters_d_x.nClusters();
    clusters_g_x = MkFitSiStripClustersCUDA(nStrips, kClusterMaxStrips, stream);
    clusters_g_x.setNClusters(nStrips);
    auto global_data_d = clusters_g_x.gview();
    const int nthreads = 128;
    const int nSeeds = std::min(MAX_SEEDSTRIPS, nStrips);
    const int nblocks = (nStrips+nthreads-1)/nthreads;
    localToGlobal<<<nblocks, nthreads, 0, stream>>>(clust_data_d, global_data_d,nStrips);
    cudaCheck(cudaGetLastError());
  }

}
