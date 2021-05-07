#ifndef _LOCAL_TO_GLOBAL_GPU_KERNEL_
#define _LOCAL_TO_GLOBAL_GPU_KERNEL_

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

static constexpr int DETS_barrel = 7932;
static constexpr int DETS_endcap = 7216;

namespace stripgpu {
  __host__ __device__ int index_lookup3(const unsigned int detid);
  __host__ __device__ int index_lookup4(const unsigned int detid);
  __host__ __device__ int index_lookup5(const unsigned int detid);
  __host__ __device__ int index_lookup6(const unsigned int detid);
}

struct LocalToGlobalMap {
  short indexer3[31214];
  short indexer5[46426];
  short indexer4[16208];
  short indexer6[145652];
  
  int det_num_bd[DETS_barrel];
  double pitch_bd[DETS_barrel];
  double offset_bd[DETS_barrel];
  double len_bd[DETS_barrel];
  double pos_x_bd[DETS_barrel];
  double pos_y_bd[DETS_barrel];
  double pos_z_bd[DETS_barrel];
  double R11_bd[DETS_barrel];
  double R12_bd[DETS_barrel];
  double R13_bd[DETS_barrel];
  double R21_bd[DETS_barrel];
  double R22_bd[DETS_barrel];
  double R23_bd[DETS_barrel];
  double backPlane_bd[DETS_barrel];
  double thickness_bd[DETS_barrel];
  double drift_x_bd[DETS_barrel];
  
  int det_num_ed[DETS_endcap];
  int yAx_ed[DETS_endcap];
  double rCross_ed[DETS_endcap];
  double aw_ed[DETS_endcap];
  double phi_ed[DETS_endcap];
  double len_ed[DETS_endcap];
  double pos_x_ed[DETS_endcap];
  double pos_y_ed[DETS_endcap];
  double pos_z_ed[DETS_endcap];
  double R11_ed[DETS_endcap];
  double R12_ed[DETS_endcap];
  double R13_ed[DETS_endcap];
  double R21_ed[DETS_endcap];
  double R22_ed[DETS_endcap];
  double R23_ed[DETS_endcap];
  double backPlane_ed[DETS_endcap];
  double thickness_ed[DETS_endcap];
  double drift_x_ed[DETS_endcap];
  double drift_y_ed[DETS_endcap];
};
#endif
