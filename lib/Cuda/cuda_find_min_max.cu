#include "include/global_const.h"
//#include "cuda_reduce.h"
//#include "cuda_dslash_ops.h"
//#include "cuda_err.cuh"

// vec*=f_aux_dev
__global__ void RescaleKernel(float2 *vec)
  {
  float2 aux1, aux2, aux3;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  aux1=vec[idx       ];
  aux2=vec[idx  +size];
  aux3=vec[idx+2*size];

  aux1.x*=f_aux_dev;
  aux1.y*=f_aux_dev;
  aux2.x*=f_aux_dev;
  aux2.y*=f_aux_dev;
  aux3.x*=f_aux_dev;
  aux3.y*=f_aux_dev;

  vec[idx       ]=aux1;
  vec[idx  +size]=aux2;
  vec[idx+2*size]=aux3;
  }


// p-> f_aux_dev*p-MMp
__global__ void SubtractKernel(float2 *p, float2 *MMp)
  {
  float2 a1, a2, a3;
  float2 b1, b2, b3;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  a1=p[idx       ];
  a2=p[idx  +size];
  a3=p[idx+2*size];

  a1.x*=f_aux_dev;
  a1.y*=f_aux_dev;
  a2.x*=f_aux_dev;
  a2.y*=f_aux_dev;
  a3.x*=f_aux_dev;
  a3.y*=f_aux_dev;

  b1=MMp[idx       ];
  b2=MMp[idx  +size];
  b3=MMp[idx+2*size];

  a1.x-=b1.x;
  a1.y-=b1.y;
  a2.x-=b2.x;
  a2.y-=b2.y;
  a3.x-=b3.x;
  a3.y-=b3.y;

  p[idx       ]=a1;
  p[idx  +size]=a2;
  p[idx+2*size]=a3;
  }


/*
============================================================== EXTERNAL C FUNCTION
*/


extern "C" void cuda_find_max(REAL *max)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_find_max...\n");
  #endif

  int loop_count=0;
  double *norm_dev;
  double norm_host, old_norm, inv_norm;
  float norm_f;
  float2 *p, *Mp;

  size_t vector_size = sizeof(float2)*3*size ;

#ifdef DEBUG_MODE_2
  double norm_check = 0;
  for (long int k=0; k<size; k++) {
    norm_check +=  simple_fermion_packed[k]*simple_fermion_packed[k];
  }
  printf("DEBUG_2: [cuda_find_max] Norm_check^2 : %f\n", norm_check); 
  if (norm_check < 10e-10 )   exit(1);
#endif 

  cudaSafe(AT,cudaMalloc((void**)&norm_dev, sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&p  , vector_size), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&Mp , vector_size), "cudaMalloc");

  cudaSafe(AT,cudaMemset(p, 0, vector_size), "cudaMemset");
  cudaSafe(AT,cudaMemcpy(p,        simple_fermion_packed,        size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(p+size,   simple_fermion_packed+size,   size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(p+2*size, simple_fermion_packed+2*size, size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

  dim3 BlockL(NUM_THREADS);
  dim3 GridL(size/(BlockL.x));

  Norm2(norm_dev, p);

  cudaSafe(AT,cudaMemcpy(&norm_host, norm_dev, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
#ifdef DEBUG_MODE_2
  printf("DEBUG_2: [cuda_find_max] Norm_host^2 : %f\n", norm_host); 
  if (norm_host < 10e-10 )   exit(1);
#endif 
  norm_host=sqrt(norm_host);
  inv_norm=1.0/norm_host;
  norm_f=(float) inv_norm;

  do{
    cudaSafe(AT,cudaMemcpyToSymbol(f_aux_dev, &norm_f, sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
    RescaleKernel<<<GridL, BlockL>>>(p);
    cudaCheckError(AT,"RescaleKernel");
    old_norm=norm_host;

    DslashOperatorEO(Mp, p, PLUS);
    DslashOperatorEO(p, Mp, MINUS);

    Norm2(norm_dev, p);
    cudaSafe(AT,cudaMemcpy(&norm_host, norm_dev,   sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    norm_host=sqrt(norm_host);
    inv_norm=1.0/norm_host;
    norm_f=(float) inv_norm;

    old_norm=fabs(old_norm-norm_host);
    old_norm/=norm_host;
    loop_count++;
    } while(old_norm>5.0e-4);

  *max=(REAL) norm_host;

  cudaSafe(AT,cudaFree(norm_dev), "cudaFree");
  cudaSafe(AT,cudaFree(p), "cudaFree");
  cudaSafe(AT,cudaFree(Mp), "cudaFree");

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_find_max in %d iterations\n", loop_count);
  #endif
  }




extern "C" void cuda_find_min(REAL *min, const REAL max)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_find_min...\n");
  #endif


  int loop_count=0;
  double *norm_dev;
  double norm_host, old_norm, inv_norm;
  float norm_f, max_f, max_d;
  float2 *p, *Mp, *MMp;

  max_f=(float) max;
  max_d=(double) max;
  size_t vector_size = sizeof(float2)*3*size ;

  cudaSafe(AT,cudaMalloc((void**)&norm_dev, sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&p   , vector_size), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&Mp  , vector_size), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&MMp , vector_size), "cudaMallox");

  cudaSafe(AT,cudaMemset(p, 0, vector_size), "cudaMemset");
  cudaSafe(AT,cudaMemcpy(p,        simple_fermion_packed,        size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(p+size,   simple_fermion_packed+size,   size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(p+2*size, simple_fermion_packed+2*size, size*sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy");

  dim3 BlockL(NUM_THREADS);
  dim3 GridL(size/(BlockL.x));

  Norm2(norm_dev, p);
  cudaSafe(AT,cudaMemcpy(&norm_host, norm_dev,   sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
  #ifdef DEBUG_MODE_2
  printf("DEBUG_2: [cuda_find_min] Norm_host^2 : %f\n", norm_host); 
  #endif 
  norm_host=sqrt(norm_host);
  inv_norm=1.0/norm_host;
  norm_f=(float) inv_norm;

  do{
    cudaSafe(AT,cudaMemcpyToSymbol(f_aux_dev, &norm_f, sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
    RescaleKernel<<<GridL, BlockL>>>(p);
    cudaCheckError(AT,"RescaleKernel");
    old_norm=norm_host;

    DslashOperatorEO(Mp, p, PLUS);
    DslashOperatorEO(MMp, Mp, MINUS);

    cudaSafe(AT,cudaMemcpyToSymbol(f_aux_dev, &max_f, sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
    SubtractKernel<<<GridL, BlockL>>>(p, MMp);
    cudaCheckError(AT,"SubtractKernel");

    Norm2(norm_dev, p);
    cudaSafe(AT,cudaMemcpy(&norm_host, norm_dev,   sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    norm_host=sqrt(norm_host);
    inv_norm=1.0/norm_host;
    norm_f=(float) inv_norm;

    old_norm=fabs(old_norm-norm_host);
    old_norm/=fabs(max_d-norm_host);
    loop_count++;
    } while(old_norm>5.0e-4);

  *min=(REAL) (max_d-norm_host);

  cudaSafe(AT,cudaFree(norm_dev), "cudaFree");
  cudaSafe(AT,cudaFree(p), "cudaFree");
  cudaSafe(AT,cudaFree(Mp), "cudaFree");
  cudaSafe(AT,cudaFree(MMp), "cudaFree");

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_find_min in %d iterations\n", loop_count);
  #endif
  }
