#ifndef CUDA_RHMC_H_
#define CUDA_RHMC_H_

#include <stdio.h>
#include <math.h>    

#include "include/global_macro.h"
#include "include/parameters.h"
#include "include/global_const.h"
#include "cuda_err.cuh"

#ifdef CUDA_LOCAL
#define CUDA_EXTERN
#else
#define CUDA_EXTERN extern
#endif

//#define USE_PINNED
//#define USE_INTRINSIC

#define PLUS   1
#define MINUS -1

#define DeclareMatrixRegs  float4 link0, link1, link2

#define LoadLinkRegs(gauge, vol, idx, mu)	\
  link0 = tex1Dfetch(gauge, idx+vol*(0+3*mu));	\
  link1 = tex1Dfetch(gauge, idx+vol*(1+3*mu));	\
  link2 = tex1Dfetch(gauge, idx+vol*(2+3*mu))

// to be used in Dirac matrix third row reconstruction
#define C1RE ( (link0.z*link2.z-link0.w*link2.w) - (link1.x*link2.x-link1.y*link2.y) )
#define C1IM (-(link0.z*link2.w+link0.w*link2.z) + (link1.x*link2.y+link1.y*link2.x) )
#define C2RE ( (link1.x*link1.z-link1.y*link1.w) - (link0.x*link2.z-link0.y*link2.w) )
#define C2IM (-(link1.x*link1.w+link1.y*link1.z) + (link0.x*link2.w+link0.y*link2.z) )
#define C3RE ( (link0.x*link2.x-link0.y*link2.y) - (link0.z*link1.z-link0.w*link1.w) )
#define C3IM (-(link0.x*link2.y+link0.y*link2.x) + (link0.z*link1.w+link0.w*link1.z) )

// same as before but in double precision
#define C1RED ( (link0z*link2z-link0w*link2w) - (link1x*link2x-link1y*link2y) )
#define C1IMD (-(link0z*link2w+link0w*link2z) + (link1x*link2y+link1y*link2x) )
#define C2RED ( (link1x*link1z-link1y*link1w) - (link0x*link2z-link0y*link2w) )
#define C2IMD (-(link1x*link1w+link1y*link1z) + (link0x*link2w+link0y*link2z) )
#define C3RED ( (link0x*link2x-link0y*link2y) - (link0z*link1z-link0w*link1w) )
#define C3IMD (-(link0x*link2y+link0y*link2x) + (link0z*link1w+link0w*link1z) )

__device__ __constant__ int size_dev;
__device__ __constant__ int size_dev_h;
__device__ __constant__ int num_shifts_device;

__device__ __constant__ float mass_dev;
__device__ __constant__ double mass_d_dev;

__device__ __constant__ float f_aux_dev;  // generic auxiliary float number
__device__ __constant__ double d_aux_dev;  // generic auxiliary double number

__device__ __constant__ float shift_coeff[max_approx_order];
__device__ __constant__ double shift_coeff_d[max_approx_order];

__device__ __constant__ float residuals[max_approx_order];
__device__ __constant__ double residuals_d[max_approx_order];

#ifdef IM_CHEM_POT
__device__ __constant__ float dev_eim_cos_f;
__device__ __constant__ float dev_eim_sin_f;
__device__ __constant__ double dev_eim_cos_d;
__device__ __constant__ double dev_eim_sin_d;
#endif

texture<float4, 1, cudaReadModeElementType> gauge_texRef;
texture<float2, 1, cudaReadModeElementType> fermion_texRef;

CUDA_EXTERN float4 *gauge_field_device;// 12*no_links elements
CUDA_EXTERN float2 *mf_device;         //  3*no_ps*size elements = no_ps fermions   ---> multiple fermion
CUDA_EXTERN float2 *smf_device;        //  3*max_approx_order*no_ps*size elements = no_ps*max_approx_order fermions ->shift multi fermion
CUDA_EXTERN float4 *ipdot_device;      //  8*no_links elements 
CUDA_EXTERN float4 *momenta_device;    //  8*no_links elements 
CUDA_EXTERN int *device_table;         //  8*size elements
CUDA_EXTERN int *device_phases;        //  4*size elements

// On host
extern float *simple_fermion_packed;
extern float *chi_packed;
extern float *psi_packed;
extern float *gauge_field_packed;
extern float *ipdot_packed;
extern float *momenta_packed;
extern int *shift_table;
extern int *eta;

void CuDoe(float2 *out, 
           float2 *in); 


#endif //CUDA_RHMC_H_

