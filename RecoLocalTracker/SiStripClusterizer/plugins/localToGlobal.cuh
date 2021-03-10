#ifndef _LOCAL_TO_GLOBAL_GPU_KERNEL_
#define _LOCAL_TO_GLOBAL_GPU_KERNEL_

#include "CUDADataFormats/SiStripCluster/interface/MkFitSiStripClustersCUDA.h"

#include <cstdint>

static constexpr int DETS_barrel = 7932;
static constexpr int DETS_endcap = 7216;

__device__ short indexer3[31214];
__device__ short indexer5[46426];
__device__ short indexer4[16208];
__device__ short indexer6[145652];

__device__ int det_num_bd[DETS_barrel];
__device__ double pitch_bd[DETS_barrel];
__device__ double offset_bd[DETS_barrel];
__device__ double len_bd[DETS_barrel];
__device__ double pos_x_bd[DETS_barrel];
__device__ double pos_y_bd[DETS_barrel];
__device__ double pos_z_bd[DETS_barrel];
__device__ double R11_bd[DETS_barrel];
__device__ double R12_bd[DETS_barrel];
__device__ double R13_bd[DETS_barrel];
__device__ double R21_bd[DETS_barrel];
__device__ double R22_bd[DETS_barrel];
__device__ double R23_bd[DETS_barrel];
__device__ double backPlane_bd[DETS_barrel];
__device__ double thickness_bd[DETS_barrel];
__device__ double drift_x_bd[DETS_barrel];

__device__ int det_num_ed[DETS_endcap];
__device__ int yAx_ed[DETS_endcap];
__device__ double rCross_ed[DETS_endcap];
__device__ double aw_ed[DETS_endcap];
__device__ double phi_ed[DETS_endcap];
__device__ double len_ed[DETS_endcap];
__device__ double pos_x_ed[DETS_endcap];
__device__ double pos_y_ed[DETS_endcap];
__device__ double pos_z_ed[DETS_endcap];
__device__ double R11_ed[DETS_endcap];
__device__ double R12_ed[DETS_endcap];
__device__ double R13_ed[DETS_endcap];
__device__ double R21_ed[DETS_endcap];
__device__ double R22_ed[DETS_endcap];
__device__ double R23_ed[DETS_endcap];
__device__ double backPlane_ed[DETS_endcap];
__device__ double thickness_ed[DETS_endcap];
__device__ double drift_x_ed[DETS_endcap];
__device__ double drift_y_ed[DETS_endcap];
#endif
