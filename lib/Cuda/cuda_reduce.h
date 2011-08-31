#ifndef CUDA_REDUCE_H_
#define CUDA_REDUCE_H_


void Norm2(double *d, float2 *vector) ;
void Norm2D(double *d, double2 *vector) ;

template <unsigned int blockSize>__global__ void ReduceSingleDGPU(double *input, double *output) ;
template <unsigned int blockSize>__global__ void ReduceDGPU(double *input, double *output) ;
template <unsigned int blockSize>__global__ void ReduceGPU(float *input, float *output) ;

void Reduce(float *in, float *out, int blocks, int threads, int sharedmem) ;
void ReduceDouble(double *in, double *out, int blocks, int threads, int sharedmem) ;

void IpdotNorm2(double *d, float4 *ipdot);

#endif

