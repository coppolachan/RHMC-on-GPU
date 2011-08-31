#include "include/global_const.h"


__global__ void NormKernel(double *d,
                           size_t ferm_offset,
			   float2 *vec) 
  {

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int grid_length = blockDim.x * gridDim.x;

  __shared__ double norm[128];  //Allocates shared mem

  float2 f_0, f_1, f_2;  
 
  norm[threadIdx.x] = 1.0;

  //First block of sites

   if(idx < size_dev)  
     {
       f_0 = tex1Dfetch(fermion_texRef,              idx + ferm_offset);
       f_1 = tex1Dfetch(fermion_texRef,   size_dev + idx + ferm_offset);
       f_2 = tex1Dfetch(fermion_texRef, 2*size_dev + idx + ferm_offset);
    
 #ifdef USE_INTRINSIC  
       norm[threadIdx.x]  = __dmul_rn((double)f_0.x, (double)f_0.x);
       norm[threadIdx.x] += __dmul_rn((double)f_0.y, (double)f_0.y);
       norm[threadIdx.x] += __dmul_rn((double)f_1.x, (double)f_1.x);
       norm[threadIdx.x] += __dmul_rn((double)f_1.y, (double)f_1.y);
       norm[threadIdx.x] += __dmul_rn((double)f_2.x, (double)f_2.x);
       norm[threadIdx.x] += __dmul_rn((double)f_2.y, (double)f_2.y);
 #else
       norm[threadIdx.x] = (double)f_0.x*(double)f_0.x+(double)f_0.y*(double)f_0.y+
       	(double)f_1.x*(double)f_1.x+(double)f_1.y*(double)f_1.y+
       	(double)f_2.x*(double)f_2.x+(double)f_2.y*(double)f_2.y;
 #endif   
      
       idx += grid_length;
     }
  
   //Other blocks of sites
   while (idx < size_dev) 
     {
       f_0 = tex1Dfetch(fermion_texRef,              idx + ferm_offset);
       f_1 = tex1Dfetch(fermion_texRef,   size_dev + idx + ferm_offset);
       f_2 = tex1Dfetch(fermion_texRef, 2*size_dev + idx + ferm_offset);
      
 #ifdef USE_INTRINSIC
       norm[threadIdx.x] += __dmul_rn((double)f_0.x, (double)f_0.x);
       norm[threadIdx.x] += __dmul_rn((double)f_0.y, (double)f_0.y);
       norm[threadIdx.x] += __dmul_rn((double)f_1.x, (double)f_1.x);
       norm[threadIdx.x] += __dmul_rn((double)f_1.y, (double)f_1.y);
       norm[threadIdx.x] += __dmul_rn((double)f_2.x, (double)f_2.x);
       norm[threadIdx.x] += __dmul_rn((double)f_2.y, (double)f_2.y);
 #else
       norm[threadIdx.x] += (double)f_0.x*(double)f_0.x+(double)f_0.y*(double)f_0.y+
       	(double)f_1.x*(double)f_1.x+(double)f_1.y*(double)f_1.y+
       	(double)f_2.x*(double)f_2.x+(double)f_2.y*(double)f_2.y;
 #endif    
     
       idx += grid_length;
     }
  __syncthreads();
  
  //Performs first reduction
  if (threadIdx.x < 64) 
    { 
      norm[threadIdx.x] += norm[threadIdx.x+64]; 
    }
  __syncthreads();
  
  if (threadIdx.x < 32 )   //Inside a warp - no syncthreads() needed
    {
      volatile double *smem = norm;

      smem[threadIdx.x] += smem[threadIdx.x + 32]; 
      smem[threadIdx.x] += smem[threadIdx.x + 16]; 
      smem[threadIdx.x] += smem[threadIdx.x +  8]; 
      smem[threadIdx.x] += smem[threadIdx.x +  4]; 
      smem[threadIdx.x] += smem[threadIdx.x +  2]; 
      smem[threadIdx.x] += smem[threadIdx.x +  1];
    }

  if (threadIdx.x == 0) d[blockIdx.x] = norm[0];
  //Outputs gridDim.x numbers to be further reduced
  }

__global__ void NormKernelD(double *d,
                            double2 *vec) 
  {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int grid_length = blockDim.x * gridDim.x;

  __shared__ double norm[128];  //Allocates shared mem

  double2 f_0, f_1, f_2;  

  norm[threadIdx.x] = 0.0;

  //First block of sites
  if(idx < size_dev)  
    {
    f_0 = vec[              idx];
    f_1 = vec[   size_dev + idx];
    f_2 = vec[ 2*size_dev + idx];

    #ifdef USE_INTRINSIC  
    norm[threadIdx.x]  = __dmul_rn(f_0.x, f_0.x);
    norm[threadIdx.x] += __dmul_rn(f_0.y, f_0.y);
    norm[threadIdx.x] += __dmul_rn(f_1.x, f_1.x);
    norm[threadIdx.x] += __dmul_rn(f_1.y, f_1.y);
    norm[threadIdx.x] += __dmul_rn(f_2.x, f_2.x);
    norm[threadIdx.x] += __dmul_rn(f_2.y, f_2.y);
    #else
    norm[threadIdx.x] = f_0.x*f_0.x+f_0.y*f_0.y+
      f_1.x*f_1.x+f_1.y*f_1.y+
      f_2.x*f_2.x+f_2.y*f_2.y;
    #endif   
 
    idx += grid_length;
    }

  //Other blocks of sites
  while (idx < size_dev) 
    {
    f_0 = vec[              idx];
    f_1 = vec[   size_dev + idx];
    f_2 = vec[ 2*size_dev + idx];

    #ifdef USE_INTRINSIC  
    norm[threadIdx.x] += __dmul_rn(f_0.x, f_0.x);
    norm[threadIdx.x] += __dmul_rn(f_0.y, f_0.y);
    norm[threadIdx.x] += __dmul_rn(f_1.x, f_1.x);
    norm[threadIdx.x] += __dmul_rn(f_1.y, f_1.y);
    norm[threadIdx.x] += __dmul_rn(f_2.x, f_2.x);
    norm[threadIdx.x] += __dmul_rn(f_2.y, f_2.y);
    #else
    norm[threadIdx.x] += f_0.x*f_0.x+f_0.y*f_0.y+
      f_1.x*f_1.x+f_1.y*f_1.y+
      f_2.x*f_2.x+f_2.y*f_2.y;
    #endif   

    idx += grid_length;
    }
  __syncthreads();

  //Performs first reduction
  if (threadIdx.x < 64) 
     { 
     norm[threadIdx.x] += norm[threadIdx.x+64]; 
     }
  __syncthreads();

  if (threadIdx.x < 32 )   //Inside a warp - no syncthreads() needed
    {
     volatile double *smem = norm;

      smem[threadIdx.x] += smem[threadIdx.x + 32]; 
      smem[threadIdx.x] += smem[threadIdx.x + 16]; 
      smem[threadIdx.x] += smem[threadIdx.x +  8]; 
      smem[threadIdx.x] += smem[threadIdx.x +  4]; 
      smem[threadIdx.x] += smem[threadIdx.x +  2]; 
      smem[threadIdx.x] += smem[threadIdx.x +  1];
    }

  if (threadIdx.x == 0) d[blockIdx.x] = norm[0];
  //Outputs gridDim.x numbers to be further reduced
  }





__global__ void IpdotNormKernel(double *d,
                                float4 *ipdot) 
  {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int grid_length = blockDim.x * gridDim.x;

  __shared__ double norm[128];  //Allocates shared mem

  float4 f_0;  

  norm[threadIdx.x] = 0.0;

  //First block of sites
  if(idx < 8*size_dev)  
    {
    f_0 = ipdot[idx];

    #ifdef USE_INTRINSIC  
    norm[threadIdx.x]  = __dmul_rn((double)f_0.x, (double)f_0.x);
    norm[threadIdx.x] += __dmul_rn((double)f_0.y, (double)f_0.y);
    norm[threadIdx.x] += __dmul_rn((double)f_0.z, (double)f_0.z);
    norm[threadIdx.x] += __dmul_rn((double)f_0.w, (double)f_0.w);
    #else
    norm[threadIdx.x] = (double)f_0.x*(double)f_0.x+(double)f_0.y*(double)f_0.y+
      (double)f_0.z*(double)f_0.z+(double)f_0.w*(double)f_0.w;
    #endif   
 
    idx += grid_length;
    }

  //Other blocks of sites
  while (idx < 8*size_dev) 
    {
    f_0 = ipdot[idx];

    #ifdef USE_INTRINSIC  
    norm[threadIdx.x] += __dmul_rn((double)f_0.x, (double)f_0.x);
    norm[threadIdx.x] += __dmul_rn((double)f_0.y, (double)f_0.y);
    norm[threadIdx.x] += __dmul_rn((double)f_0.z, (double)f_0.z);
    norm[threadIdx.x] += __dmul_rn((double)f_0.w, (double)f_0.w);
    #else
    norm[threadIdx.x] += (double)f_0.x*(double)f_0.x+(double)f_0.y*(double)f_0.y+
      (double)f_0.z*(double)f_0.z+(double)f_0.w*(double)f_0.w;
    #endif   
 
    idx += grid_length;
    }

  __syncthreads();

  //Performs first reduction
  if (threadIdx.x < 64) 
     { 
     norm[threadIdx.x] += norm[threadIdx.x+64]; 
     }
  __syncthreads();

  if (threadIdx.x < 32 )   //Inside a warp - no syncthreads() needed
    {
     volatile double *smem = norm;

      smem[threadIdx.x] += smem[threadIdx.x + 32]; 
      smem[threadIdx.x] += smem[threadIdx.x + 16]; 
      smem[threadIdx.x] += smem[threadIdx.x +  8]; 
      smem[threadIdx.x] += smem[threadIdx.x +  4]; 
      smem[threadIdx.x] += smem[threadIdx.x +  2]; 
      smem[threadIdx.x] += smem[threadIdx.x +  1];
    }

  if (threadIdx.x == 0) d[blockIdx.x] = norm[0];
  //Outputs gridDim.x numbers to be further reduced
  }


template <unsigned int blockSize>__global__ void ReduceGPU(float *input, float *output) 
  {
  __shared__ float sdata[blockSize];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int ix  = blockIdx.x * (blockSize*2) + threadIdx.x;
  sdata[tid] = input[ix]+ input[ix + blockSize];
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

  if(tid < 32)
    {
        volatile float * smem = sdata;
        if (blockSize >=  64) { smem[tid] += smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] += smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] += smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] += smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] += smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] += smem[tid +  1]; }

	// if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; }
	// if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; }
	// if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; }
	// if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; }
	// if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; }
	// if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }

  // write result for this block to global mem
  if (tid == 0) output[blockIdx.x] = sdata[0];
  }


template <unsigned int blockSize>__global__ void ReduceDGPU(double *input, double *output) 
  {
   __shared__ double sdata[blockSize];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int ix  = blockIdx.x * (blockSize*2) + threadIdx.x;
  sdata[tid] = input[ix]+ input[ix + blockSize];
  __syncthreads();

  // do reduction in shared mem
  if(blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
  if(blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
  if(blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

  if(tid < 32)
    {
        volatile double * smem = sdata;
        if (blockSize >=  64) { smem[tid] += smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] += smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] += smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] += smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] += smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] += smem[tid +  1]; }
    }

  // write result for this block to global mem
  if (tid == 0) output[blockIdx.x] = sdata[0];
  }






template <unsigned int blockSize>__global__ void ReduceSingleDGPU(double *input, double *output) 
  {
  __shared__ double sdata[blockSize];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  sdata[tid] = input[tid];
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >=  64) { if (threadIdx.x < 32 && threadIdx.x + 32 < blockDim.x) { sdata[tid] += sdata[tid + 32]; } __syncthreads();}
  if (blockSize >=  32) { if (threadIdx.x < 16 && threadIdx.x + 16 < blockDim.x) { sdata[tid] += sdata[tid + 16]; } __syncthreads();}
  if (blockSize >=  16) { if (threadIdx.x < 8 && threadIdx.x + 8 < blockDim.x) { sdata[tid] += sdata[tid +  8]; } __syncthreads();}
  if (blockSize >=   8) { if (threadIdx.x < 4 && threadIdx.x + 4 < blockDim.x) { sdata[tid] += sdata[tid +  4]; } __syncthreads();}
  if (blockSize >=   4) { if (threadIdx.x < 2 && threadIdx.x + 2 < blockDim.x) { sdata[tid] += sdata[tid +  2]; } __syncthreads();}
  if (blockSize >=   2) { if (threadIdx.x < 1 && threadIdx.x + 1 < blockDim.x) { sdata[tid] += sdata[tid +  1]; } }
  
  // write result for this block to global mem
  if (tid == 0) output[blockIdx.x] = sdata[0];
  }






void Reduce(float *in, float *out, int blocks, int threads, int sharedmem) 
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside Reduce ...\33[0m\n");
  #endif

  switch(threads)
    {
    case 512:
      ReduceGPU<512><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<512>");
      break;
    case 256:
      ReduceGPU<256><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<256>");
      break;
    case 128:
      ReduceGPU<128><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<128>");
      break;
    case 64:
      ReduceGPU<64><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<64>");
      break;
    case 32:
      ReduceGPU<32><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<32>");
      break;
    case 16:
      ReduceGPU<16><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<16>");
      break;
    case 8:
      ReduceGPU<8><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<8>");
      break;
    case 4:
      ReduceGPU<4><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<4>");
      break;
    case 2:
      ReduceGPU<2><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<2>");
      break;
    case 1:
      ReduceGPU<1><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceGPU<1>");
      break;
    }

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated Reduce\033[0m\n");
  #endif
  }





void ReduceDouble(double *in, double *out, int blocks, int threads, int sharedmem) 
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside ReduceDouble ...\033[0m\n");
  #endif

  switch(threads)
    {
    case 512:
      ReduceDGPU<512><<<blocks, threads, sharedmem>>>(in, out);
      cudaCheckError(AT,"ReduceDGPU<512>");
      break;
    case 256:
      ReduceDGPU<256><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<256>");
      break;
    case 128:
      ReduceDGPU<128><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<128>");
      break;
    case 64:
      ReduceDGPU<64><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<64>");
      break;
    case 32:
      ReduceDGPU<32><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<32>");
      break;
    case 16:
      ReduceDGPU<16><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<16>");
      break;
    case 8:
      ReduceDGPU<8><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<8>");
      break;
    case 4:
      ReduceDGPU<4><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<4>");
      break;
    case 2:
      ReduceDGPU<2><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<2>");
      break;
    case 1:
      ReduceDGPU<1><<<blocks, threads, sharedmem>>>(in, out); 
      cudaCheckError(AT,"ReduceDGPU<1>");
      break;
    }

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated ReduceDouble\033[0m\n");
  #endif
  }





void Norm2(double *d, float2 *vector) 
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside Norm2 ...\033[0m\n");
  #endif

  //Create and destroy here the variables for accumulation
  //They need to be of dimension NormGrid.x
  unsigned int threads, sharedmem; 
  
  dim3 NormBlock(128);  // here 128 is needed (see NormKernel) 
  const unsigned int grid_size_limit = 1 << (int)ceil(log2((double)size/(double)NormBlock.x));  //Number of blocks 
  const unsigned int grid_size = (grid_size_limit < 64) ? grid_size_limit : 64;
  dim3 NormGrid(grid_size); 

  size_t vector_size=3*size*sizeof(float2);
#ifdef DEBUG_MODE_2
  float2 loc_vec;
  cudaSafe(AT,cudaMemcpy(&loc_vec, vector, sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  printf("\033[32mDEBUG: Input vector size : %d\033[0m\n",(int)(vector_size));
  printf("\033[32mDEBUG: Parameters : Grid size %d\033[0m\n",grid_size);
  printf("\033[32mDEBUG: Vector[0]  : %f , %f \033[0m\n",loc_vec.x, loc_vec.y); 
#endif
  double *temp_d;
  cudaSafe(AT,cudaMalloc((void**)&temp_d, sizeof(double)*grid_size), "cudaMalloc");

  size_t offset_f;
  cudaSafe(AT,cudaBindTexture(&offset_f, fermion_texRef, vector, vector_size), "cudaBindTexture");
  offset_f/=sizeof(float2);

  NormKernel<<<NormGrid, NormBlock>>>(temp_d, offset_f, vector);
  cudaCheckError(AT,"NormKernel");
  //Accumulates moving a window of grid_size elements
  //Outputs a vector of grid_size elements (<=64 and power of 2)

#ifdef DEBUG_MODE_2
  double local_t[64];
  int k;
  cudaSafe(AT,cudaMemcpy(&local_t, temp_d, grid_size*sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
  for (k = 0; k < grid_size; k++) {
    printf("\033[32m\tTemporary vector [%d] : %f\033[0m\n", k, local_t[k]);
  }
#endif
  
  cudaSafe(AT,cudaUnbindTexture(fermion_texRef), "cudaUnbindTexture");
  
  //Further reduction  
  threads = NormGrid.x;
  sharedmem  = threads*sizeof(double);
  switch(threads) 
    {
    case 64:
      ReduceSingleDGPU<64><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<64>");
      break;
    case 32:
      ReduceSingleDGPU<32><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<32>");
      break;
    case 16:
      ReduceSingleDGPU<16><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<16>");
      break;
    case 8:
      ReduceSingleDGPU<8><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<8>");
      break;
    case 4:
      ReduceSingleDGPU<4><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<4>");
      break;
    case 2:
      ReduceSingleDGPU<2><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<2>");
      break;
    case 1:
      ReduceSingleDGPU<1><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<1>");
      break;
    }
 

  cudaSafe(AT,cudaMemcpy(d, temp_d, sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaFree(temp_d), "cudaFree");




  #ifdef DEBUG_MODE_2
  double local;
  cudaSafe(AT,cudaMemcpy(&local, d, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
  printf("\033[32m\tNorm2 result : %f\033[0m\n", local);
  printf("\033[32m\tterminated Norm2 \033[0m\n");
  exit(1);
  #endif
  }


void Norm2D(double *d, double2 *vector) 
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside Norm2D ...\033[0m\n");
  #endif

  //Create and destroy here the variables for accumulation
  //They need to be of dimension NormGrid.x
  unsigned int threads, sharedmem; 
  
  dim3 NormBlock(128);  // here 128 is needed (see NormKernel) 
  const unsigned int grid_size_limit = 1 << (int)ceil(log2((double)size/(double)NormBlock.x));  //Number of blocks 
  const unsigned int grid_size = (grid_size_limit < 64) ? grid_size_limit : 64;
  dim3 NormGrid(grid_size); 

  double *temp_d;
  cudaSafe(AT,cudaMalloc((void**)&temp_d, sizeof(double)*grid_size), "cudaMalloc");

  NormKernelD<<<NormGrid, NormBlock>>>(temp_d, vector);
  cudaCheckError(AT,"NormKernelD");
  //Accumulates moving a window of grid_size elements
  //Outputs a vector of grid_size elements (<=64 and power of 2)

  //Further reduction  
  threads = NormGrid.x;
  sharedmem  = threads*sizeof(double);
  switch(threads) 
    {
    case 64:
      ReduceSingleDGPU<64><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<64>");
      break;
    case 32:
      ReduceSingleDGPU<32><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<32>");
      break;
    case 16:
      ReduceSingleDGPU<16><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<16>");
      break;
    case 8:
      ReduceSingleDGPU<8><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<8>");
      break;
    case 4:
      ReduceSingleDGPU<4><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<4>");
      break;
    case 2:
      ReduceSingleDGPU<2><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<2>");
      break;
    case 1:
      ReduceSingleDGPU<1><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<1>");
      break;
    }

  cudaSafe(AT,cudaMemcpy(d, temp_d, sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaFree(temp_d), "cudaFree");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated Norm2D \033[0m\n");
  #endif
  }





void IpdotNorm2(double *d, float4 *ipdot)   // ipdot has 2*no_links=8*size float4 elements 
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside IpdotNorm2 ...\033[0m\n");
  #endif

  //Create and destroy here the variables for accumulation
  //They need to be of dimension NormGrid.x
  unsigned int threads, sharedmem; 
  
  dim3 NormBlock(128);  // here 128 is needed (see NormKernel) 
  const unsigned int grid_size_limit = 1 << (int)ceil(log2((8.0*(double)size)/(double)NormBlock.x));  //Number of blocks 
  const unsigned int grid_size = (grid_size_limit < 64) ? grid_size_limit : 64;
  dim3 NormGrid(grid_size); 

  double *temp_d;
  cudaSafe(AT,cudaMalloc((void**)&temp_d, sizeof(double)*grid_size), "cudaMalloc");

  IpdotNormKernel<<<NormGrid, NormBlock>>>(temp_d, ipdot);
  cudaCheckError(AT,"IpdotNormKernel");
  //Accumulates moving a window of grid_size elements
  //Outputs a vector of grid_size elements (<=64 and power of 2)

  //Further reduction  
  threads = NormGrid.x;
  sharedmem  = threads*sizeof(double);
  switch(threads) 
    {
    case 64:
      ReduceSingleDGPU<64><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<64>");
      break;
    case 32:
      ReduceSingleDGPU<32><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<32>");
      break;
    case 16:
      ReduceSingleDGPU<16><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<16>");
      break;
    case 8:
      ReduceSingleDGPU<8><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<8>");
      break;
    case 4:
      ReduceSingleDGPU<4><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<4>");
      break;
    case 2:
      ReduceSingleDGPU<2><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<2>");
      break;
    case 1:
      ReduceSingleDGPU<1><<<1, threads, sharedmem>>>(temp_d, temp_d); 
      cudaCheckError(AT,"ReduceSingleDGPU<1>");
      break;
    }

  cudaSafe(AT,cudaMemcpy(d, temp_d, sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaFree(temp_d), "cudaFree");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated IpdotNorm2 \033[0m\n");
  #endif
  }

