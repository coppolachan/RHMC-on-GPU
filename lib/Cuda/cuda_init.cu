#include "include/global_const.h"
#define CUDA_LOCAL
#include "cuda_rhmc.cuh"

extern "C" void cuda_init0(void)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_init0 ...\n");
  #endif

  size_t gauge_field_size_f = 3*sizeof(float4)*no_links;    //first two lines only of each SU(3) matrix

  // allocate & initialize gauge
  // 2 since 1double~2float
  #ifdef USE_PINNED
  float *gauge_pinned;
  cudaSafe(AT,cudaHostAlloc((void**)&gauge_pinned, 2*gauge_field_size_f, cudaHostAllocWriteCombined), "cudaHostAlloc");
  cudaSafe(AT,cudaMemcpy(gauge_pinned, gauge_field_packed, 2*gauge_field_size_f, cudaMemcpyHostToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMalloc((void**)&gauge_field_device, 2*gauge_field_size_f), "cudaMalloc"); 
  cudaSafe(AT,cudaMemcpy(gauge_field_device, gauge_pinned, 2*gauge_field_size_f, cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaFreeHost(gauge_pinned), "cudaFreeHost");
  #else
  cudaSafe(AT,cudaMalloc((void**)&gauge_field_device, 2*gauge_field_size_f), "cudaMalloc");
  cudaSafe(AT,cudaMemcpy(gauge_field_device, gauge_field_packed, 2*gauge_field_size_f, cudaMemcpyHostToDevice), 
                 "cudaMemcpy");
  #endif

  // allocate & initialize device_table
  cudaSafe(AT,cudaMalloc((void**)&device_table, sizeof(int)*size*8), "cudaMalloc");
  cudaSafe(AT,cudaMemcpy(device_table, shift_table, sizeof(int)*size*8, cudaMemcpyHostToDevice), "cudaMemcpy"); 

  // allocate & initialize device_phases
  cudaSafe(AT,cudaMalloc((void**)&device_phases, sizeof(int)*size*4), "cudaMalloc");
  cudaSafe(AT,cudaMemcpy(device_phases, eta, sizeof(int)*size*4, cudaMemcpyHostToDevice), "cudaMemcpy"); 


  // initialize constants
  float mass_l=(float) GlobalParams::Instance().getMass();
  double mass = GlobalParams::Instance().getMass();
  cudaSafe(AT,cudaMemcpyToSymbol(mass_dev, &mass_l, sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
  cudaSafe(AT,cudaMemcpyToSymbol(mass_d_dev, &mass, sizeof(double), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
  int size_l=(int) size;
  cudaSafe(AT,cudaMemcpyToSymbol(size_dev, &size_l, sizeof(int), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
  size_l=(int) sizeh;
  cudaSafe(AT,cudaMemcpyToSymbol(size_dev_h, &size_l, sizeof(int), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");

  if (GlobalChemPotPar::Instance().UseChem()) {
    double eim_cos = GlobalChemPotPar::Instance().getEim_cos();
    double eim_sin = GlobalChemPotPar::Instance().getEim_sin();
    mass_l=(float)eim_cos;
    cudaSafe(AT,cudaMemcpyToSymbol(dev_eim_cos_f, &mass_l, sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
    mass_l=(float)eim_sin;
    cudaSafe(AT,cudaMemcpyToSymbol(dev_eim_sin_f, &mass_l, sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
    cudaSafe(AT,cudaMemcpyToSymbol(dev_eim_cos_d, &eim_cos, sizeof(double), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
    cudaSafe(AT,cudaMemcpyToSymbol(dev_eim_sin_d, &eim_sin, sizeof(double), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
  }

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_init0\n");
  #endif
  }


extern "C" void cuda_init1(void)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_init1 ...\n");
  #endif

  size_t vector_size_f   = sizeof(float2)*3*size;           // 2(complex)*3(su3_vector)

  // allocate & initialize mf_device
  // again 2 since 1double~2float
  cudaSafe(AT,cudaMalloc((void**)&mf_device, 2*no_ps*vector_size_f), "cudaMalloc"); 
  cudaSafe(AT,cudaMemset(mf_device, 0, 2*no_ps*vector_size_f), "cudaMemset");  // initialize even and odd to 0
  for(int ps=0; ps<no_ps; ps++)                  // copy the even entries from host
     {
     // 1st float
     cudaSafe(AT,cudaMemcpy(mf_device          + ps*3*size, chi_packed          + ps*3*size, 
                                    size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(mf_device +   size + ps*3*size, chi_packed +   size + ps*3*size, 
                                    size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(mf_device + 2*size + ps*3*size, chi_packed + 2*size + ps*3*size, 
                                    size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

     // 2nd float
     cudaSafe(AT,cudaMemcpy(mf_device          + ps*3*size + no_ps*3*size, chi_packed          + ps*3*size + no_ps*3*size, 
                                    size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(mf_device +   size + ps*3*size + no_ps*3*size, chi_packed +   size + ps*3*size + no_ps*3*size, 
                                    size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(mf_device + 2*size + ps*3*size + no_ps*3*size, chi_packed + 2*size + ps*3*size + no_ps*3*size, 
                                    size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

     }

  // allocate & initialize to zero smf_device (even & odd)
  // again 2 since 1double~2float
  cudaSafe(AT,cudaMalloc((void**)&smf_device, 2*no_ps*max_approx_order*vector_size_f), "cudaMalloc"); 
  cudaSafe(AT,cudaMemset(smf_device, 0, 2*no_ps*max_approx_order*vector_size_f), "cudaMemset"); 

  // allocate & initialize ipdot_device
  cudaSafe(AT,cudaMalloc((void**)&ipdot_device, 8*no_links*sizeof(float)), "cudaMalloc"); 
  cudaSafe(AT,cudaMemcpy(ipdot_device, ipdot_packed, 8*no_links*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy"); 

  // allocate & initialize momenta_device
  cudaSafe(AT,cudaMalloc((void**)&momenta_device, 8*no_links*sizeof(float)), "cudaMalloc"); 
  cudaSafe(AT,cudaMemcpy(momenta_device, momenta_packed, 8*no_links*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_init1\n");
  #endif
  }


extern "C" void cuda_meas_init(void)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_meas_init ...\n");
  #endif

  size_t vector_size_f   = sizeof(float2)*3*size;           // 2(complex)*3(su3_vector)

  // allocate & initialize to zero mf_device
  // again 2 since 1double~2float
  cudaSafe(AT,cudaMalloc((void**)&mf_device, 2*no_ps*vector_size_f), "cudaMalloc"); 
  cudaSafe(AT,cudaMemset(mf_device, 0, 2*no_ps*vector_size_f), "cudaMemset");  // initialize even and odd to 0

  // allocate & initialize to zero smf_device (even & odd)
  // again 2 since 1double~2float
  cudaSafe(AT,cudaMalloc((void**)&smf_device, 2*no_ps*max_approx_order*vector_size_f), "cudaMalloc"); 
  cudaSafe(AT,cudaMemset(smf_device, 0, 2*no_ps*max_approx_order*vector_size_f), "cudaMemset"); 

  // 1st float
  cudaSafe(AT,cudaMemcpy(mf_device         , simple_fermion_packed         , 
                                 size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(mf_device +   size, simple_fermion_packed +   size, 
                                 size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(mf_device + 2*size, simple_fermion_packed + 2*size, 
                                 size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

  // 2nd float
  cudaSafe(AT,cudaMemcpy(mf_device         + no_ps*3*size, simple_fermion_packed           + 3*size, 
                                 size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(mf_device +  size + no_ps*3*size, simple_fermion_packed +    size + 3*size, 
                                 size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(mf_device + 2*size + no_ps*3*size, simple_fermion_packed + 2*size + 3*size, 
                                 size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");


  #ifdef DEBUG_MODE
  printf("\tterminated cuda_meas_init\n");
  #endif
  }


extern "C" void cuda_meas_end(void)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_meas_end ...\n");
  #endif

  const int offset_2f=3*size*1*no_ps;  // num_shifts=1

  // 1st float
  cudaSafe(AT,cudaMemcpy(simple_fermion_packed,          smf_device, 
                                     sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(simple_fermion_packed +   size, smf_device +   size, 
                                     sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(simple_fermion_packed + 2*size, smf_device + 2*size, 
                                     sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  // 2nd float
  cudaSafe(AT,cudaMemcpy(simple_fermion_packed +          3*size, smf_device +          offset_2f, 
                                     sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(simple_fermion_packed +   size + 3*size, smf_device +   size + offset_2f, 
                                     sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(simple_fermion_packed + 2*size + 3*size, smf_device + 2*size + offset_2f, 
                                     sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");

  cudaSafe(AT,cudaFree(mf_device), "cudaFree");
  cudaSafe(AT,cudaFree(smf_device), "cudaFree");

 
  #ifdef DEBUG_MODE
  printf("\tterminated cuda_meas_end\n");
  #endif
  }



extern "C" void cuda_end(void)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_end ...\n");
  #endif

  cudaSafe(AT,cudaMemcpy(gauge_field_packed, gauge_field_device, 2*3*no_links*sizeof(float4), cudaMemcpyDeviceToHost), 
                                                                                                            "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(momenta_packed, momenta_device, 8*no_links*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(ipdot_packed, ipdot_device, sizeof(float)*no_links*8, cudaMemcpyDeviceToHost), "cudaMemcpy");

  cudaSafe(AT,cudaFree(gauge_field_device), "cudaFree");
  cudaSafe(AT,cudaFree(device_table), "cudaFree");
  cudaSafe(AT,cudaFree(device_phases), "cudaFree");

  cudaSafe(AT,cudaFree(mf_device), "cudaFree");
  cudaSafe(AT,cudaFree(smf_device), "cudaFree");

  cudaSafe(AT,cudaFree(ipdot_device), "cudaFree");
  cudaSafe(AT,cudaFree(momenta_device), "cudaFree");

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_end\n");
  #endif
  }


extern "C" void cuda_get_conf(void)
 {
 #ifdef DEBUG_MODE
 printf("DEBUG: inside cuda_get_conf ...\n");
 #endif

 cudaSafe(AT,cudaMemcpy(gauge_field_packed, gauge_field_device, 2*12*no_links*sizeof(float), cudaMemcpyDeviceToHost), 
       "cudaMemcpy");

 #ifdef DEBUG_MODE
 printf("\tterminated cuda_get_conf ...\n");
 #endif
 }


extern "C" void cuda_get_momenta(void)
 {
 #ifdef DEBUG_MODE
 printf("DEBUG: inside cuda_get_momenta ...\n");
 #endif

 cudaSafe(AT,cudaMemcpy(momenta_packed, momenta_device, 8*no_links*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");

 #ifdef DEBUG_MODE
 printf("\tterminated cuda_get_momenta ...\n");
 #endif
 }


extern "C" void cuda_put_momenta(void)
 {
 #ifdef DEBUG_MODE
 printf("DEBUG: inside cuda_put_momenta ...\n");
 #endif

 cudaSafe(AT,cudaMemcpy(momenta_device, momenta_packed, 8*no_links*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

 #ifdef DEBUG_MODE
 printf("\tterminated cuda_put_momenta ...\n");
 #endif
 }


// Include all kernels

#include "cuda_tool_kernels.h"

#include "cuda_dslash_kernels.h"

#include "cuda_inversion_kernels.h"

#include "cuda_update_kernels.h"


