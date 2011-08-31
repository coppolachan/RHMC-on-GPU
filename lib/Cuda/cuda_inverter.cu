
__global__ void CalculateNewResidualKernel(double *b,                    // beta
			                   double *prev_res,             // previous residue
			                   double *d,                    // <p,A.p>
			                   double *res,                  // temporary residues (to be reduced)
			                   float2 *residual_vect,        // residue vector
			                   float2 *A_p)                  // A.p
  {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tid;
  const unsigned int grid_size = blockDim.x * gridDim.x;
  double b_double;

  __shared__ float bsh;
  __shared__ double norm[128];
  norm[threadIdx.x] = 0.0;

  if( tid == 0 ) 
    {
    b_double = -(*prev_res)/(*d);
    bsh=(float)b_double;
    if (blockIdx.x == 0) (*b) = b_double;
    }
  __syncthreads();  //bsh visible to all threads

  float2 res1, res2, res3, Ap1, Ap2, Ap3;
  float2 out1, out2, out3;
 
  if(idx < size_dev)  
    {
    res1 = residual_vect[idx               ];
    res2 = residual_vect[idx +     size_dev];
    res3 = residual_vect[idx + 2 * size_dev];
    
    Ap1 = A_p[idx               ];
    Ap2 = A_p[idx +     size_dev];
    Ap3 = A_p[idx + 2 * size_dev];
  
    #ifdef USE_INTRINSIC 
    out1.x = __fmaf_rn(bsh, Ap1.x, res1.x);
    out1.y = __fmaf_rn(bsh, Ap1.y, res1.y);
    out2.x = __fmaf_rn(bsh, Ap2.x, res2.x);
    out2.y = __fmaf_rn(bsh, Ap2.y, res2.y);
    out3.x = __fmaf_rn(bsh, Ap3.x, res3.x);
    out3.y = __fmaf_rn(bsh, Ap3.y, res3.y);
    #else
    out1.x = res1.x + bsh * Ap1.x;
    out1.y = res1.y + bsh * Ap1.y;
    out2.x = res2.x + bsh * Ap2.x;
    out2.y = res2.y + bsh * Ap2.y;
    out3.x = res3.x + bsh * Ap3.x;
    out3.y = res3.y + bsh * Ap3.y;
    #endif   
 
    #ifdef USE_INTRINSIC
    norm[threadIdx.x]  = __dmul_rn((double)out1.x, (double)out1.x);
    norm[threadIdx.x] += __dmul_rn((double)out1.y, (double)out1.y);
    norm[threadIdx.x] += __dmul_rn((double)out2.x, (double)out2.x);
    norm[threadIdx.x] += __dmul_rn((double)out2.y, (double)out2.y);
    norm[threadIdx.x] += __dmul_rn((double)out3.x, (double)out3.x);
    norm[threadIdx.x] += __dmul_rn((double)out3.y, (double)out3.y);
    #else
    norm[threadIdx.x] = (double)out1.x*(double)out1.x + (double)out1.y*(double)out1.y+
      (double)out2.x*(double)out2.x + (double)out2.y*(double)out2.y+
      (double)out3.x*(double)out3.x + (double)out3.y*(double)out3.y;
    #endif
    
    residual_vect[idx               ] = out1;
    residual_vect[idx +     size_dev] = out2;
    residual_vect[idx + 2 * size_dev] = out3;
    
    idx += grid_size;
    }

  //Other blocks of sites
  while(idx < size_dev) 
    {
    res1 = residual_vect[idx               ];
    res2 = residual_vect[idx +     size_dev];
    res3 = residual_vect[idx + 2 * size_dev];
    
    Ap1 = A_p[idx               ];
    Ap2 = A_p[idx +     size_dev];
    Ap3 = A_p[idx + 2 * size_dev];

    #ifdef USE_INTRINSIC
    out1.x = __fmaf_rn(bsh, Ap1.x, res1.x);
    out1.y = __fmaf_rn(bsh, Ap1.y, res1.y);
    out2.x = __fmaf_rn(bsh, Ap2.x, res2.x);
    out2.y = __fmaf_rn(bsh, Ap2.y, res2.y);
    out3.x = __fmaf_rn(bsh, Ap3.x, res3.x);
    out3.y = __fmaf_rn(bsh, Ap3.y, res3.y);
    #else
    out1.x = res1.x + bsh * Ap1.x;
    out1.y = res1.y + bsh * Ap1.y;
    out2.x = res2.x + bsh * Ap2.x;
    out2.y = res2.y + bsh * Ap2.y;
    out3.x = res3.x + bsh * Ap3.x;
    out3.y = res3.y + bsh * Ap3.y;
    #endif
  
    #ifdef USE_INTRINSIC
    norm[threadIdx.x] += __dmul_rn((double)out1.x, (double)out1.x);
    norm[threadIdx.x] += __dmul_rn((double)out1.y, (double)out1.y);
    norm[threadIdx.x] += __dmul_rn((double)out2.x, (double)out2.x);
    norm[threadIdx.x] += __dmul_rn((double)out2.y, (double)out2.y);
    norm[threadIdx.x] += __dmul_rn((double)out3.x, (double)out3.x);
    norm[threadIdx.x] += __dmul_rn((double)out3.y, (double)out3.y);
    #else
    norm[threadIdx.x] += (double)out1.x*(double)out1.x + (double)out1.y*(double)out1.y+
      (double)out2.x*(double)out2.x + (double)out2.y*(double)out2.y+
      (double)out3.x*(double)out3.x + (double)out3.y*(double)out3.y;
    #endif

    residual_vect[idx               ] = out1;
    residual_vect[idx +     size_dev] = out2;
    residual_vect[idx + 2 * size_dev] = out3;
  
    idx += grid_size;
    }
  __syncthreads();

  //Performs first reduction
  if (threadIdx.x < 64) { norm[threadIdx.x] += norm[threadIdx.x + 64]; }
  __syncthreads();

  if (threadIdx.x < 32 )   //Inside a warp - no syncthreads() needed
    {
      volatile double *smem = norm;
      // norm[threadIdx.x] += norm[threadIdx.x + 32]; 
      // norm[threadIdx.x] += norm[threadIdx.x + 16]; 
      // norm[threadIdx.x] += norm[threadIdx.x +  8]; 
      // norm[threadIdx.x] += norm[threadIdx.x +  4]; 
      // norm[threadIdx.x] += norm[threadIdx.x +  2]; 
      // norm[threadIdx.x] += norm[threadIdx.x +  1]; 

      smem[threadIdx.x] += smem[threadIdx.x + 32]; 
      smem[threadIdx.x] += smem[threadIdx.x + 16]; 
      smem[threadIdx.x] += smem[threadIdx.x +  8]; 
      smem[threadIdx.x] += smem[threadIdx.x +  4]; 
      smem[threadIdx.x] += smem[threadIdx.x +  2]; 
      smem[threadIdx.x] += smem[threadIdx.x +  1]; 
    }
 
  if (threadIdx.x == 0) res[blockIdx.x] = norm[0];
  //Outputs gridDim.x numbers to be further reduced
  }




__global__ void InitializeShiftVarsKernel(double  *bs, 
			                  double  *z, 
			                  double  *b, 
			                  float2 *chi,
			                  float2 *psi)
  {
  // blockIdx.y = sigma

  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // as initialized idx<sizeh
  double z1, z2, bs_double;

  __shared__ float bs_sh;  //Used in vector sum

  float2 temp_chi1, temp_chi2, temp_chi3, out1, out2, out3;

  if(threadIdx.x == 0) 
    {
                                                                    // zeta_-1=1.0
    z1    = 1.0;                                                    // zeta_0
    z2    = 1.0/(1.0 - (double)shift_coeff[blockIdx.y] * (*b) );    // zeta_1             (*b=beta_0, beta_-1=1.0)
    bs_double = (*b) * z2;  // beta^{sigma}_0
    bs_sh = (float)bs_double;
    if(blockIdx.x == 0) 
      {
      //Safely write on global memory
      z[                    blockIdx.y] = z1;  // zeta_0
      z[num_shifts_device + blockIdx.y] = z2;  // zeta_1
      bs[blockIdx.y] = bs_double;
      }
    }

  __syncthreads();  //bs_sh accessible to all threads

  // psi^{sigma}_1-=beta^sigma}_0*chi     (p^{sigma}_0=chi)

  // even sites
  temp_chi1 = chi[idx             ];
  temp_chi2 = chi[idx +   size_dev];
  temp_chi3 = chi[idx + 2*size_dev];

  #ifdef USE_INTRINSIC
  out1.x = __fmul_rn(- bs_sh, temp_chi1.x);
  out1.y = __fmul_rn(- bs_sh, temp_chi1.y);
  out2.x = __fmul_rn(- bs_sh, temp_chi2.x);
  out2.y = __fmul_rn(- bs_sh, temp_chi2.y);
  out3.x = __fmul_rn(- bs_sh, temp_chi3.x);
  out3.y = __fmul_rn(- bs_sh, temp_chi3.y);
  #else
  out1.x = - bs_sh * temp_chi1.x;
  out1.y = - bs_sh * temp_chi1.y;
  out2.x = - bs_sh * temp_chi2.x;
  out2.y = - bs_sh * temp_chi2.y;
  out3.x = - bs_sh * temp_chi3.x;
  out3.y = - bs_sh * temp_chi3.y;
  #endif

  psi[(3*blockIdx.y    )*size_dev + idx] = out1;
  psi[(3*blockIdx.y + 1)*size_dev + idx] = out2;
  psi[(3*blockIdx.y + 2)*size_dev + idx] = out3;
  }





__global__ void UpdatePsKernel(double *a, 
			       double *c, 
			       double *cp, 
			       double *z, 
			       double *bs, 
			       double *b, 
			       float2 *p_0, 
			       float2 *r, 
			       float2 *p,
			       int *converged,
			       int *iz)
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double a_double, as_double;

  __shared__ float a_sh, as;
  __shared__ int convBlock;
  float z_th;

  if(threadIdx.x == 0) 
    {
    a_double = (*c)/(*cp);
    a_sh = (float)a_double;
    convBlock = converged[blockIdx.y];
    if(!convBlock) 
      {
      as_double = a_double * z[(*iz)*num_shifts_device + blockIdx.y]*bs[blockIdx.y]/
	( z[ (1-(*iz)) * num_shifts_device + blockIdx.y] * (*b) );
      
      #ifdef USE_INTRINSIC
      as = __double2float_rn(as_double);
      #else 
      as=(float)as_double;
      #endif
      }
    if (blockIdx.x==0 && blockIdx.y == 0) (*a) = a_double;
    }

  __syncthreads();

  float2 res1, res2, res3;
  float2 p_1, p_2, p_3;
  float2 out1, out2, out3; 

  res1 = r[idx             ];
  res2 = r[idx +   size_dev];
  res3 = r[idx + 2*size_dev];

  if(blockIdx.y==0) 
    {
    p_1 = p_0[idx             ];
    p_2 = p_0[idx +   size_dev];
    p_3 = p_0[idx + 2*size_dev];

    #ifdef USE_INTRINSIC
    out1.x = __fmaf_rn(a_sh, p_1.x, res1.x);
    out1.y = __fmaf_rn(a_sh, p_1.y, res1.y);
    out2.x = __fmaf_rn(a_sh, p_2.x, res2.x);
    out2.y = __fmaf_rn(a_sh, p_2.y, res2.y);
    out3.x = __fmaf_rn(a_sh, p_3.x, res3.x);
    out3.y = __fmaf_rn(a_sh, p_3.y, res3.y);
    #else
    out1.x = res1.x + a_sh * p_1.x;
    out1.y = res1.y + a_sh * p_1.y;
    out2.x = res2.x + a_sh * p_2.x;
    out2.y = res2.y + a_sh * p_2.y;
    out3.x = res3.x + a_sh * p_3.x;
    out3.y = res3.y + a_sh * p_3.y;
    #endif

    p_0[idx             ] = out1;
    p_0[idx +   size_dev] = out2;
    p_0[idx + 2*size_dev] = out3;
    } 
 
  if( !convBlock )
    {
    p_1 = p[(3*blockIdx.y    )*size_dev + idx];
    p_2 = p[(3*blockIdx.y + 1)*size_dev + idx];
    p_3 = p[(3*blockIdx.y + 2)*size_dev + idx];

    #ifdef USE_INTRINSIC
    z_th = __double2float_rz(z[(*iz)*num_shifts_device + blockIdx.y]);
    #else
    z_th = (float) z[(*iz)*num_shifts_device + blockIdx.y];
    #endif

    #ifdef USE_INTRINSIC
    float out_temp;    
    out_temp = __fmul_rn(z_th, res1.x);
    out1.x   = __fmaf_rn(as,    p_1.x, out_temp);
    out_temp = __fmul_rn(z_th, res1.y);
    out1.y   = __fmaf_rn(as,    p_1.y, out_temp);
    out_temp = __fmul_rn(z_th, res2.x);
    out2.x   = __fmaf_rn(as,    p_2.x, out_temp);
    out_temp = __fmul_rn(z_th, res2.y);
    out2.y   = __fmaf_rn(as,    p_2.y, out_temp);
    out_temp = __fmul_rn(z_th, res3.x);
    out3.x   = __fmaf_rn(as,    p_3.x, out_temp);
    out_temp = __fmul_rn(z_th, res3.y);
    out3.y   = __fmaf_rn(as,    p_3.y, out_temp);
    #else
    out1.x = z_th * res1.x + as * p_1.x;
    out1.y = z_th * res1.y + as * p_1.y;
    out2.x = z_th * res2.x + as * p_2.x;
    out2.y = z_th * res2.y + as * p_2.y;
    out3.x = z_th * res3.x + as * p_3.x;
    out3.y = z_th * res3.y + as * p_3.y;
    #endif

    p[(3*blockIdx.y    )*size_dev + idx] = out1;
    p[(3*blockIdx.y + 1)*size_dev + idx] = out2;
    p[(3*blockIdx.y + 2)*size_dev + idx] = out3;
    }
  }






__global__ void NewShiftVarsKernel(double  *bs, 
			           double  *bp,
			           double  *z, 
			           double  *z_out, 
			           double  *b,
			           double  *a,
			           double  *c, 
			           float2 *psi,
			           float2 *p,
			           int *converged,
			           int *iz)
  {
  double z0, z1, ztmp;
  double css, bs_double;
  __shared__ float bs_sh;  //Used in vector sum
  __shared__ int convBlock;
  float2 temp_psi1, temp_psi2, temp_psi3, out1, out2, out3;
  float2 p1, p2, p3;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  // blockIdx.y=sigma
  if(threadIdx.x == 0) 
    {
    convBlock = converged[blockIdx.y];
    if(!convBlock) 
      {
      z0    = z[ (1-(*iz)) * num_shifts_device + blockIdx.y];             //z0=zeta_n^{sigma}
      z1    = z[     (*iz) * num_shifts_device + blockIdx.y];             //z1=zeta_{n-1}^sigma
      ztmp  = (z0 * z1 * (*bp)) /( (*b) * (*a) * (z1 - z0) + z1 * 
				 (*bp)*(1.0 - (double)shift_coeff[blockIdx.y] * (*b) ));   // ztmp=zeta_{n+1}^{sigma}

      // bs_double=beta_n^{sigma}
      bs_double = (*b) * ztmp/z0; // since res^{sigma}_{i+1}=zeta^{sigma}_{i+1}*res_{i+1} and *c=(res_{i+1}, res_{i+1})


      #ifdef USE_INTRINSIC
      bs_sh = __double2float_rz(bs_double);
      #else
      bs_sh=(float)(bs_double);
      #endif

      css = (*c) * ztmp * ztmp;
     
      if(blockIdx.x == 0) 
        {
	if (css < residuals[blockIdx.y])
	  converged[blockIdx.y] = 1;
	//Safely write on global memory
	z_out[ (1-(*iz)) * num_shifts_device + blockIdx.y] = z0;
	z_out[ (*iz) * num_shifts_device + blockIdx.y] = ztmp;	
	bs[blockIdx.y] = bs_double;
        }
      }
    }

  __syncthreads();  //bs_sh and convBlock accessible to all threads

  if(!convBlock) 
    {
    // even sites
    temp_psi1 = psi[(3*blockIdx.y    )*size_dev + idx];
    temp_psi2 = psi[(3*blockIdx.y + 1)*size_dev + idx];
    temp_psi3 = psi[(3*blockIdx.y + 2)*size_dev + idx];
    
    p1 = p[(3*blockIdx.y    )*size_dev + idx];
    p2 = p[(3*blockIdx.y + 1)*size_dev + idx];
    p3 = p[(3*blockIdx.y + 2)*size_dev + idx];

    #ifdef USE_INTRINSIC   
    out1.x = __fmul_rn(-bs_sh, p1.x);
    out1.x += temp_psi1.x;
    out1.y = __fmul_rn(-bs_sh, p1.y);
    out1.y += temp_psi1.y;
    out2.x = __fmul_rn(-bs_sh, p2.x);
    out2.x += temp_psi2.x;
    out2.y = __fmul_rn(-bs_sh, p2.y);
    out2.y += temp_psi2.y;
    out3.x = __fmul_rn(-bs_sh, p3.x);
    out3.x += temp_psi3.x;
    out3.y = __fmul_rn(-bs_sh, p3.y);
    out3.y += temp_psi3.y;
    #else 
    out1.x = temp_psi1.x - bs_sh * p1.x;
    out1.y = temp_psi1.y - bs_sh * p1.y;
    out2.x = temp_psi2.x - bs_sh * p2.x;
    out2.y = temp_psi2.y - bs_sh * p2.y;
    out3.x = temp_psi3.x - bs_sh * p3.x;
    out3.y = temp_psi3.y - bs_sh * p3.y;
    #endif

    psi[(3*blockIdx.y    )*size_dev + idx] = out1;
    psi[(3*blockIdx.y + 1)*size_dev + idx] = out2;
    psi[(3*blockIdx.y + 2)*size_dev + idx] = out3;
    }
  }


/*
================================================================================== END OF KERNELS
*/


void CalculateNewResidual(double* prev_res,      // previous residual
			  double* res,           // residual
			  double* b_prev,        // previous beta
			  double* b,             // beta
			  double* d,             // d=<p,A.p>
			  float2* residual_vect, // residual vector
			  float2* A_p)           // A.p
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside CalculateNewResidual ...\033[0m\n");
  #endif

  // prev_res  = cp  ;  res = c ; residual_vect = r ; A_p = MMp

  unsigned int threads, sharedmem; 
  
  dim3 ResBlock(128);  // must be 128, see CalcResidual
  const unsigned int grid_size_limit = 1 << (int)ceil(log2((double)size/(2.0 * (double)ResBlock.x)));  //Number of blocks 
  const unsigned int grid_size = (grid_size_limit < 64) ? grid_size_limit : 64;
  dim3 ResGrid(grid_size);

  double *temp_res;
  cudaSafe(AT,cudaMalloc((void**)&temp_res, sizeof(double)*grid_size), "cudaMalloc");

  cudaSafe(AT,cudaMemcpy(prev_res, res, sizeof(double), cudaMemcpyDeviceToDevice), "cudaMalloc");
  cudaSafe(AT,cudaMemcpy(  b_prev,   b, sizeof(double), cudaMemcpyDeviceToDevice), "cudaMalloc");

  CalculateNewResidualKernel<<<ResGrid, ResBlock>>>(b, prev_res, d, temp_res, residual_vect, A_p);
  cudaCheckError(AT,"CalculateNewResidualKernel");
  //Accumulates moving a window of grid_size elements
  //Outputs a vector of grid_size elements (<=64 and power of 2)

  //Further reduction  
  threads = ResGrid.x;
  sharedmem  = threads*sizeof(double);
  switch(threads) 
    {
    case 64:
      ReduceSingleDGPU<64><<<1, threads, sharedmem>>>(temp_res, temp_res); 
      cudaCheckError(AT,"ReduceSingleDGPU<64>");
      break;
    case 32:
      ReduceSingleDGPU<32><<<1, threads, sharedmem>>>(temp_res, temp_res); 
      cudaCheckError(AT,"ReduceSingleDGPU<32>");
      break;
    case 16:
      ReduceSingleDGPU<16><<<1, threads, sharedmem>>>(temp_res, temp_res); 
      cudaCheckError(AT,"ReduceSingleDGPU<16>");
      break;
    case 8:
      ReduceSingleDGPU<8><<<1, threads, sharedmem>>>(temp_res, temp_res); 
      cudaCheckError(AT,"ReduceSingleDGPU<8>");
      break;
    case 4:
      ReduceSingleDGPU<4><<<1, threads, sharedmem>>>(temp_res, temp_res); 
      cudaCheckError(AT,"ReduceSingleDGPU<4>");
      break;
    case 2:
      ReduceSingleDGPU<2><<<1, threads, sharedmem>>>(temp_res, temp_res); 
      cudaCheckError(AT,"ReduceSingleDGPU<2>");
      break;
    case 1:
      ReduceSingleDGPU<1><<<1, threads, sharedmem>>>(temp_res, temp_res); 
      cudaCheckError(AT,"ReduceSingleDGPU<1>");
      break;
    }

  cudaSafe(AT,cudaMemcpy(res, temp_res, sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy");

  cudaSafe(AT,cudaFree(temp_res), "cudaFree");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated CalculateNewResidual\033[0m\n");
  #endif
  }




void InitializeShiftVars(double *bs, 
			 double *z, 
			 double *b,
			 float2 *chi_dev, 
			 float2 *psi_dev,
			 const int num_shifts)
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside InitializeShiftVars ...\033[0m\n");
  #endif

  dim3 InitShiftsBlock(NUM_THREADS);
  dim3 InitShiftsGrid(size/(2*InitShiftsBlock.x), num_shifts);

  InitializeShiftVarsKernel<<<InitShiftsGrid, InitShiftsBlock>>>(bs, z, b, chi_dev, psi_dev);
  cudaCheckError(AT,"InitializeShiftVarsKernel");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated InitializeShiftVars\033[0m\n");
  #endif
  }





void UpdatePs(double *a, 
	      double *c, 
	      double *cp, 
	      double *z, 
	      double *bs, 
	      double *b, 
	      float2 *p_0, 
	      float2 *r, 
	      float2 *p,
	      int *conv,
	      int *iz,
	      const int num_shifts)
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside UpdatePs ...\033[0m\n");
  #endif

  dim3 UpdatePsBlock(NUM_THREADS);
  dim3 UpdatePsGrid(size/(2*UpdatePsBlock.x), num_shifts);

  UpdatePsKernel<<<UpdatePsGrid, UpdatePsBlock>>>(a, c, cp, z, bs, b, p_0, r, p, conv, iz);
  cudaCheckError(AT,"UpdatePsKernel");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated UpdatePs\033[0m\n");
  #endif
  }




void NewShiftVars(double *bs, 
		  double *bp,
		  double *z, 
		  double *b, 
		  double *a, 
		  double *c,
		  float2 *psi_dev,
	          float2 *p, 
		  int *conv,
		  int *iz,
		  const int num_shifts)
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside NewShiftVars ...\033[0m\n");
  #endif

  double *z_out;
  cudaSafe(AT,cudaMalloc((void**)&z_out ,2*num_shifts*sizeof(double)), "cudaSafe");

  dim3 ShiftsBlock(NUM_THREADS);
  dim3 ShiftsGrid(size/(2*ShiftsBlock.x), num_shifts);

  NewShiftVarsKernel<<<ShiftsGrid, ShiftsBlock>>>(bs, bp, z, z_out, b, a,  c, psi_dev, p, conv, iz);
  cudaCheckError(AT,"NewShiftVarsKernel");

  cudaSafe(AT,cudaMemcpy(z, z_out, 2*num_shifts*sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy");

  cudaSafe(AT,cudaFree(z_out), "cudaFree");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated NewShiftVars\033[0m\n");
  #endif
  }




void DebugPrintFloat(float *devicemem, int k) 
  {
  float hostmem;
  cudaMemcpy(&hostmem,   devicemem,   sizeof(float), cudaMemcpyDeviceToHost);
  printf("Device -- [%d] Value = %.8g\n", k, hostmem);
  }

void DebugPrintDouble(double *devicemem, int k) 
  {
  double hostmem;
  cudaMemcpy(&hostmem,   devicemem,   sizeof(double), cudaMemcpyDeviceToHost);
  printf("Device -- [%d] Value = %.16g\n", k, hostmem);
  }

void DebugPrintInt(int *devicemem, int k) 
  {
  int hostmem;
  cudaMemcpy(&hostmem,   devicemem,   sizeof(int), cudaMemcpyDeviceToHost);
  printf("Device -- [%d] Value = %d\n", k, hostmem);
  }




/*
======================================================== EXTERNAL C FUNCTION
*/



// solve (D^{dag}D+shift) smf_device=mf_device


extern "C" void cuda_shifted_inverter(const double residual, 
			              const double *shifts,
			              const int num_shifts,
                                      const int psferm,
			              int *ncount)
  {
  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  printf("DEBUG: inside cuda_shifted_inverter  ...\n");
  #endif

  int iter, k, conv;

  float2 *r;                   //residual vector 
  float2 *p_0, *Mp, *MMp, *p;  //auxiliary fermion fields
  double *a, *b, *bp, *bs, *c, *cp, *d, *z;
  int *check_conv, *iz, check_conv_host[max_approx_order];
  float res_sq[num_shifts], shifts_f[num_shifts];

  int iter_iz = 1;

  const int ps_offset=3*size*psferm;
  const int offset_2f=3*size*num_shifts*no_ps;

  size_t vector_size = sizeof(float2)*3*size ;  // 2(complex)*3(su3_vector)

  //Put everything in global memory space at this stage
  cudaSafe(AT,cudaMalloc((void**)&p          ,  num_shifts*vector_size), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&r          ,             vector_size), "cudaMalloc"); 
  cudaSafe(AT,cudaMalloc((void**)&p_0        ,             vector_size), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&Mp         ,             vector_size), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&MMp        ,             vector_size), "cudaMalloc");

  cudaSafe(AT,cudaMalloc((void**)&a          ,             sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&b          ,             sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&bp         ,             sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&bs         ,  num_shifts*sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&c          ,             sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&cp         ,             sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&d          ,             sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&z          ,2*num_shifts*sizeof(double)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&iz         ,                sizeof(int)), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&check_conv ,     num_shifts*sizeof(int)), "cudaMalloc");

  // residues squared
  for(iter = 0; iter < num_shifts; iter++)  
    {
    res_sq[iter] = (float) (residual * residual);
    }

  for(iter = 0; iter < num_shifts; iter++)  
    {
    shifts_f[iter] = (float) shifts[iter]; 
    }

  // constants
  cudaSafe(AT,cudaMemcpyToSymbol(shift_coeff, shifts_f, sizeof(float)*num_shifts, 0, cudaMemcpyHostToDevice), 
                                                                           "cudaMemcpyToSymbol");
  cudaSafe(AT,cudaMemcpyToSymbol(residuals, res_sq, sizeof(float)*num_shifts, 0, cudaMemcpyHostToDevice), 
                                                                           "cudaMemcpyToSymbol");
  cudaSafe(AT,cudaMemcpyToSymbol(num_shifts_device, &num_shifts, sizeof(int), 0,   cudaMemcpyHostToDevice), 
                                                                           "cudaMemcpyToSymbol");

  // initialize to 0 smf_device
  // 1st float(s)
  cudaSafe(AT,cudaMemset(smf_device + num_shifts*ps_offset, 0, num_shifts*vector_size), "cudaMemset");
  // 2nd float(s)
  cudaSafe(AT,cudaMemset(smf_device + num_shifts*ps_offset + offset_2f, 0, num_shifts*vector_size), "cudaMemset"); 

  // initialize check counters
  cudaSafe(AT,cudaMemset(check_conv, 0, num_shifts*sizeof(int)), "cudaMemset");
  cudaSafe(AT,cudaMemcpy(iz, &iter_iz, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");

  // c=||chi||^2
  Norm2(c, mf_device + ps_offset);

  // r[0] := p[0] := chi 
  cudaSafe(AT,cudaMemcpy(  r, mf_device + ps_offset, vector_size, cudaMemcpyDeviceToDevice), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(p_0, mf_device + ps_offset, vector_size, cudaMemcpyDeviceToDevice), "cudaMemcpy");

  // p[num_shifts] := chi
  for (iter = 0; iter < num_shifts; iter++)
    { 
    cudaSafe(AT,cudaMemcpy(p+3*size*iter, mf_device + ps_offset, vector_size, cudaMemcpyDeviceToDevice), "cudaMemcpy");
    }

  //  b[0] := - | r[0] |**2 / < p[0], Ap[0] > ;  A=D^{dag}D
  //  First compute  d  =  < p, A.p > 
  //  Mp=D.p_0
  DslashOperatorEO(Mp, p_0, PLUS);

  //  d =  < Mp, Mp > =<p, A.p> 
  Norm2(d, Mp);

  // MMp = D^{dag}Mp
  DslashOperatorEO(MMp, Mp, MINUS);

  //  b = -cp/d
  //  r[1] += b[0] A.p[0] 
  //  c = |r[1]|^2   
  CalculateNewResidual(cp, c, bp, b, d, r, MMp);

  //  psi[1] -= b[0] p[0]
  InitializeShiftVars(bs, z, b, mf_device + ps_offset, smf_device + num_shifts*ps_offset, num_shifts);

  conv = 0;
  for (k=1; k < max_cg && !conv ; ++k) 
   {
    //  a[k+1] := |r[k]|**2 / |r[k-1]|**2 ; 
    //  Update p
    //  p[k+1] := r[k+1] + a[k+1] p[k]; 
    //  Compute the shifted as 
    //  ps[k+1] := zs[k+1] r[k+1] + a[k+1] ps[k];
    UpdatePs(a, c, cp, z, bs, b, p_0, r, p, check_conv, iz, num_shifts);

    // Mp = D p_0
    DslashOperatorEO(Mp, p_0, PLUS);

    //  d =  < Mp, Mp > =<p, A.p> 
    Norm2(d, Mp);

    // MMp = D^{dag} Mp
    DslashOperatorEO(MMp, Mp, MINUS);

    CalculateNewResidual(cp, c, bp, b, d, r, MMp);

    iter_iz = 1-iter_iz; 
    cudaSafe(AT,cudaMemcpy(iz,  &iter_iz, sizeof(int), cudaMemcpyHostToDevice), "cudaSafe");

    NewShiftVars(bs, bp, z, b, a, c, smf_device + num_shifts*ps_offset, p, check_conv, iz, num_shifts);

    cudaSafe(AT,cudaMemcpy(check_conv_host, check_conv, num_shifts*sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");

    conv = 1;
    for(iter = 0; iter < num_shifts; iter++) 
       {
       conv &=check_conv_host[iter];  // bitwise "and" operator
       //printf("Conv[%d] = %d\n", iter, check_conv_host[iter]);
       }

    //DebugPrintDouble(c, k);
    }
  (*ncount) = k;

  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  // copy back gauge conf to host
  size_t gauge_field_size = sizeof(float4)*size*12;
  cudaSafe(AT,cudaMemcpy(gauge_field_packed, gauge_field_device, 2*gauge_field_size, cudaMemcpyDeviceToHost), "cudaMemcpy");

  // copy back solution to host
  for(k=0; k<num_shifts; k++)
     {                                                         // ps_offset=3*size*psferm
     // 1st float
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k  )*size + num_shifts*ps_offset, 
                         smf_device + (3*k  )*size + num_shifts*ps_offset, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k+1)*size + num_shifts*ps_offset, 
                         smf_device + (3*k+1)*size + num_shifts*ps_offset, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k+2)*size + num_shifts*ps_offset, 
                         smf_device + (3*k+2)*size + num_shifts*ps_offset, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     // 2nd float
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k  )*size + num_shifts*ps_offset + offset_2f, 
                         smf_device + (3*k  )*size + num_shifts*ps_offset + offset_2f, 
                                            sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k+1)*size + num_shifts*ps_offset + offset_2f, 
                         smf_device + (3*k+1)*size + num_shifts*ps_offset + offset_2f, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k+2)*size + num_shifts*ps_offset + offset_2f, 
                         smf_device + (3*k+2)*size + num_shifts*ps_offset + offset_2f, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     }

  // copy back rhs to host
  // 1st float
  cudaSafe(AT,cudaMemcpy(chi_packed + ps_offset, 
                         mf_device + ps_offset,         sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(chi_packed + size   + ps_offset, 
                         mf_device + size   + ps_offset, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(chi_packed + 2*size + ps_offset, 
                         mf_device + 2*size + ps_offset, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  // 2nd float
  cudaSafe(AT,cudaMemcpy(chi_packed + ps_offset + no_ps*3*size, 
                mf_device + ps_offset          + no_ps*3*size, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(chi_packed + ps_offset + size + no_ps*3*size, 
                mf_device + ps_offset +   size + no_ps*3*size, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(chi_packed + 2*size + no_ps*3*size, 
                mf_device + ps_offset + 2*size + no_ps*3*size, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  #endif

  cudaSafe(AT,cudaFree(p), "cudaFree");
  cudaSafe(AT,cudaFree(r), "cudaFree");
  cudaSafe(AT,cudaFree(p_0), "cudaFree");
  cudaSafe(AT,cudaFree(Mp), "cudaFree");
  cudaSafe(AT,cudaFree(MMp), "cudaFree");

  cudaSafe(AT,cudaFree(a), "cudaFree");
  cudaSafe(AT,cudaFree(b), "cudaFree");
  cudaSafe(AT,cudaFree(bp), "cudaFree");
  cudaSafe(AT,cudaFree(bs), "cudaFree");
  cudaSafe(AT,cudaFree(c), "cudaFree");
  cudaSafe(AT,cudaFree(cp), "cudaFree");
  cudaSafe(AT,cudaFree(d), "cudaFree");
  cudaSafe(AT,cudaFree(z), "cudaFree");
  cudaSafe(AT,cudaFree(iz), "cudaFree");
  cudaSafe(AT,cudaFree(check_conv), "cudaFree");

  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  printf("\tterminated cuda_shifted_inverter\n");
  #endif
  }

