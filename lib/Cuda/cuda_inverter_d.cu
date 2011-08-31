
__global__ void InitRKernel(double2 *r,
                            float2 *in,   
			    int f2_offs)                  
  {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float2 auxferm;
  double2 ferm_in_0, ferm_in_1, ferm_in_2;

  auxferm = in[              idx];
  ferm_in_0.x=(double)auxferm.x;
  ferm_in_0.y=(double)auxferm.y;
  auxferm = in[   size_dev + idx];
  ferm_in_1.x=(double)auxferm.x;
  ferm_in_1.y=(double)auxferm.y;
  auxferm = in[ 2*size_dev + idx];
  ferm_in_2.x=(double)auxferm.x;
  ferm_in_2.y=(double)auxferm.y;

  // 2nd float
  auxferm = in[              idx + f2_offs];
  ferm_in_0.x+=(double)auxferm.x;
  ferm_in_0.y+=(double)auxferm.y;
  auxferm = in[   size_dev + idx + f2_offs];
  ferm_in_1.x+=(double)auxferm.x;
  ferm_in_1.y+=(double)auxferm.y;
  auxferm = in[ 2*size_dev + idx + f2_offs];
  ferm_in_2.x+=(double)auxferm.x;
  ferm_in_2.y+=(double)auxferm.y;

  r[             idx]=ferm_in_0;
  r[  size_dev + idx]=ferm_in_1;
  r[2*size_dev + idx]=ferm_in_2;
  }




__global__ void CalculateNewResidualDKernel(double *b,                    // beta
			                    double *prev_res,             // previous residue
			                    double *d,                    // <p,A.p>
			                    double *res,                  // temporary residues (to be reduced)
			                    double2 *residual_vect,       // residue vector
			                    double2 *A_p)                 // A.p
  {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + tid;
  const unsigned int grid_size = blockDim.x * gridDim.x;
  double b_double;

  __shared__ double bsh;
  __shared__ double norm[128];
  norm[threadIdx.x] = 0.0;

  if( tid == 0 ) 
    {
    b_double = -(*prev_res)/(*d);
    bsh=b_double;
    if(blockIdx.x == 0) 
      { 
      //Safely write on global memory
      *b = b_double;
      }
    }
  __syncthreads();  //bsh visible to all threads

  double2 res1, res2, res3, Ap1, Ap2, Ap3;
  double2 out1, out2, out3;
 
  if(idx < size_dev)  
    {
    res1 = residual_vect[idx             ];
    res2 = residual_vect[idx +   size_dev];
    res3 = residual_vect[idx + 2*size_dev];
    
    Ap1 = A_p[idx             ];
    Ap2 = A_p[idx +   size_dev];
    Ap3 = A_p[idx + 2*size_dev];
 
    #ifdef USE_INTRINSIC
    out1.x = __dadd_rn(res1.x, __dmul_rn(bsh, Ap1.x));
    out1.y = __dadd_rn(res1.y, __dmul_rn(bsh, Ap1.y));
    out2.x = __dadd_rn(res2.x, __dmul_rn(bsh, Ap2.x));
    out2.y = __dadd_rn(res2.y, __dmul_rn(bsh, Ap2.y));
    out3.x = __dadd_rn(res3.x, __dmul_rn(bsh, Ap3.x));
    out3.y = __dadd_rn(res3.y, __dmul_rn(bsh, Ap3.y));
    #else 
    out1.x = res1.x + bsh * Ap1.x;
    out1.y = res1.y + bsh * Ap1.y;
    out2.x = res2.x + bsh * Ap2.x;
    out2.y = res2.y + bsh * Ap2.y;
    out3.x = res3.x + bsh * Ap3.x;
    out3.y = res3.y + bsh * Ap3.y;
    #endif 

    #ifdef USE_INTRINSIC
    norm[threadIdx.x]  = __dmul_rn(out1.x, out1.x);
    norm[threadIdx.x] += __dmul_rn(out1.y, out1.y);
    norm[threadIdx.x] += __dmul_rn(out2.x, out2.x);
    norm[threadIdx.x] += __dmul_rn(out2.y, out2.y);
    norm[threadIdx.x] += __dmul_rn(out3.x, out3.x);
    norm[threadIdx.x] += __dmul_rn(out3.y, out3.y);
    #else
    norm[threadIdx.x] = out1.x*out1.x + out1.y*out1.y+
      out2.x*out2.x + out2.y*out2.y+
      out3.x*out3.x + out3.y*out3.y;
    #endif
    
    residual_vect[idx             ] = out1;
    residual_vect[idx +   size_dev] = out2;
    residual_vect[idx + 2*size_dev] = out3;
    
    idx += grid_size;
    }

  //Other blocks of sites
  while(idx < size_dev) 
    {
    res1 = residual_vect[idx             ];
    res2 = residual_vect[idx +   size_dev];
    res3 = residual_vect[idx + 2*size_dev];
    
    Ap1 = A_p[idx             ];
    Ap2 = A_p[idx +   size_dev];
    Ap3 = A_p[idx + 2*size_dev];

    #ifdef USE_INTRINSIC
    out1.x = __dadd_rn(res1.x, __dmul_rn(bsh, Ap1.x));
    out1.y = __dadd_rn(res1.y, __dmul_rn(bsh, Ap1.y));
    out2.x = __dadd_rn(res2.x, __dmul_rn(bsh, Ap2.x));
    out2.y = __dadd_rn(res2.y, __dmul_rn(bsh, Ap2.y));
    out3.x = __dadd_rn(res3.x, __dmul_rn(bsh, Ap3.x));
    out3.y = __dadd_rn(res3.y, __dmul_rn(bsh, Ap3.y));
    #else 
    out1.x = res1.x + bsh * Ap1.x;
    out1.y = res1.y + bsh * Ap1.y;
    out2.x = res2.x + bsh * Ap2.x;
    out2.y = res2.y + bsh * Ap2.y;
    out3.x = res3.x + bsh * Ap3.x;
    out3.y = res3.y + bsh * Ap3.y;
    #endif 
  
    #ifdef USE_INTRINSIC
    norm[threadIdx.x] += __dmul_rn(out1.x, out1.x);
    norm[threadIdx.x] += __dmul_rn(out1.y, out1.y);
    norm[threadIdx.x] += __dmul_rn(out2.x, out2.x);
    norm[threadIdx.x] += __dmul_rn(out2.y, out2.y);
    norm[threadIdx.x] += __dmul_rn(out3.x, out3.x);
    norm[threadIdx.x] += __dmul_rn(out3.y, out3.y);
    #else
    norm[threadIdx.x] += out1.x*out1.x + out1.y*out1.y+
      out2.x*out2.x + out2.y*out2.y+
      out3.x*out3.x + out3.y*out3.y;
    #endif

    residual_vect[idx             ] = out1;
    residual_vect[idx +   size_dev] = out2;
    residual_vect[idx + 2*size_dev] = out3;
  
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




__global__ void InitializeShiftVarsDKernel(double *bs, 
			                   double *z, 
			                   double *b, 
			                   double2 *chi,
			                   float2 *psi,
                                           int f2_offs)
  {
  // blockIdx.y = sigma

  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // as initialized idx<sizeh, even
  double z1, z2, bs_double;

  float aux_f;
  double aux_d;

  __shared__ double bs_sh;  // used in vector sum

  double2 temp_chi1, temp_chi2, temp_chi3, out1, out2, out3;

  if(threadIdx.x == 0) 
    {
                                                              // zeta_-1=1.0
    z1    = 1.0;                                              // zeta_0
    z2    = 1.0/(1.0 - shift_coeff_d[blockIdx.y] * (*b) );    // zeta_1             (*b=beta_0, beta_-1=1.0)
    bs_double = (*b) * z2;  // beta^{sigma}_0
    bs_sh = bs_double;
    if(blockIdx.x == 0) 
      {
      //Safely write on global memory
      z[                    blockIdx.y] = z1;  // zeta_0
      z[num_shifts_device + blockIdx.y] = z2;  // zeta_1
      bs[blockIdx.y] = bs_double;
      }
    }

  __syncthreads();  //bs_sh accessible to all threads

  // psi^{sigma}_1-=beta^{sigma}_0*chi     (p^{sigma}_0=chi)

  // even sites
  temp_chi1 = chi[idx             ];
  temp_chi2 = chi[idx +   size_dev];
  temp_chi3 = chi[idx + 2*size_dev];

  #ifdef USE_INTRINSIC
  out1.x = __dmul_rn(-bs_sh, temp_chi1.x);
  out1.y = __dmul_rn(-bs_sh, temp_chi1.y);
  out2.x = __dmul_rn(-bs_sh, temp_chi2.x);
  out2.y = __dmul_rn(-bs_sh, temp_chi2.y);
  out3.x = __dmul_rn(-bs_sh, temp_chi3.x);
  out3.y = __dmul_rn(-bs_sh, temp_chi3.y);
  #else
  out1.x = - bs_sh * temp_chi1.x;
  out1.y = - bs_sh * temp_chi1.y;
  out2.x = - bs_sh * temp_chi2.x;
  out2.y = - bs_sh * temp_chi2.y;
  out3.x = - bs_sh * temp_chi3.x;
  out3.y = - bs_sh * temp_chi3.y;
  #endif

  aux_d=out1.x;
  aux_f=(float)aux_d;
  aux_d-=(double)aux_f;
  psi[(3*blockIdx.y    )*size_dev + idx].x = aux_f;
  psi[(3*blockIdx.y    )*size_dev + idx + f2_offs].x = (float)aux_d;

  aux_d=out1.y;
  aux_f=(float) aux_d;
  aux_d-=(double)aux_f;
  psi[(3*blockIdx.y    )*size_dev + idx].y = aux_f;
  psi[(3*blockIdx.y    )*size_dev + idx + f2_offs].y = (float)aux_d;

  aux_d=out2.x;
  aux_f=(float) aux_d;
  aux_d-=(double)aux_f;
  psi[(3*blockIdx.y + 1)*size_dev + idx].x = aux_f;
  psi[(3*blockIdx.y + 1)*size_dev + idx + f2_offs].x = (float)aux_d;

  aux_d=out2.y;
  aux_f=(float) aux_d;
  aux_d-=(double)aux_f;
  psi[(3*blockIdx.y + 1)*size_dev + idx].y = aux_f;
  psi[(3*blockIdx.y + 1)*size_dev + idx + f2_offs].y = (float)aux_d;

  aux_d=out3.x;
  aux_f=(float) aux_d;
  aux_d-=(double)aux_f;
  psi[(3*blockIdx.y + 2)*size_dev + idx].x = aux_f;
  psi[(3*blockIdx.y + 2)*size_dev + idx + f2_offs].x = (float)aux_d;

  aux_d=out3.y;
  aux_f=(float) aux_d;
  aux_d-=(double)aux_f;
  psi[(3*blockIdx.y + 2)*size_dev + idx].y = aux_f;
  psi[(3*blockIdx.y + 2)*size_dev + idx + f2_offs].y = (float)aux_d;
  }





__global__ void UpdatePsDKernel(double *a, 
			       double *c, 
			       double *cp, 
			       double *z, 
			       double *bs, 
			       double *b, 
			       double2 *p_0, 
			       double2 *r, 
			       double2 *p,
			       int *converged,
			       int *iz)
  {
  // blockIdx.y = sigma

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double a_double, as_double;

  __shared__ double a_sh, as;
  __shared__ int convBlock;
  double z_th;

  if(threadIdx.x == 0) 
    {
    a_double = (*c)/(*cp);
    a_sh = a_double;
    convBlock = converged[blockIdx.y];
    if(!convBlock) 
      {
      as_double = a_double * z[(*iz)*num_shifts_device + blockIdx.y]*bs[blockIdx.y]/
	( z[ (1-(*iz)) * num_shifts_device + blockIdx.y] * (*b) );
       
      as=as_double;
      }
    if(blockIdx.x==0 && blockIdx.y == 0) 
      {
      //Safely write on global memory
      *a = a_double; 
      }
    }

  __syncthreads();

  double2 res1, res2, res3;
  double2 p_1, p_2, p_3;
  double2 out1, out2, out3; 

  res1 = r[idx             ];
  res2 = r[idx +   size_dev];
  res3 = r[idx + 2*size_dev];

  if(blockIdx.y==0) 
    {
    p_1 = p_0[idx             ];
    p_2 = p_0[idx +   size_dev];
    p_3 = p_0[idx + 2*size_dev];

    #ifdef USE_INTRINSIC
    out1.x = __dadd_rn(res1.x, __dmul_rn(a_sh, p_1.x));
    out1.y = __dadd_rn(res1.y, __dmul_rn(a_sh, p_1.y));
    out2.x = __dadd_rn(res2.x, __dmul_rn(a_sh, p_2.x));
    out2.y = __dadd_rn(res2.y, __dmul_rn(a_sh, p_2.y));
    out3.x = __dadd_rn(res3.x, __dmul_rn(a_sh, p_3.x));
    out3.y = __dadd_rn(res3.y, __dmul_rn(a_sh, p_3.y));
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

    z_th = z[(*iz)*num_shifts_device + blockIdx.y];

    #ifdef USE_INTRINSIC
    out1.x = __dadd_rn(__dmul_rn(z_th, res1.x), __dmul_rn(as, p_1.x));
    out1.y = __dadd_rn(__dmul_rn(z_th, res1.y), __dmul_rn(as, p_1.y));
    out2.x = __dadd_rn(__dmul_rn(z_th, res2.x), __dmul_rn(as, p_2.x));
    out2.y = __dadd_rn(__dmul_rn(z_th, res2.y), __dmul_rn(as, p_2.y));
    out3.x = __dadd_rn(__dmul_rn(z_th, res3.x), __dmul_rn(as, p_3.x));
    out3.y = __dadd_rn(__dmul_rn(z_th, res3.y), __dmul_rn(as, p_3.y));
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






__global__ void NewShiftVarsDKernel(double *bs, 
			            double *bp,
			            double *z, 
			            double *z_out, 
			            double *b,
			            double *a,
			            double *c, 
			            float2 *psi,
                                    int f2_offs, 
			            double2 *p,
			            int *converged,
			            int *iz)
  {
  // blockIdx.y=sigma

  double z0, z1, ztmp;
  double css, bs_double;

  __shared__ double bs_sh;  //Used in vector sum
  __shared__ int convBlock;

  double2 temp_psi1, temp_psi2, temp_psi3, out1, out2, out3;
  double2 p1, p2, p3;

  double aux_d;
  double aux_f;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(threadIdx.x == 0) 
    {
    convBlock = converged[blockIdx.y];
    if(!convBlock) 
      {
      z0    = z[ (1-(*iz)) * num_shifts_device + blockIdx.y];             //z0=zeta_n^{sigma}
      z1    = z[     (*iz) * num_shifts_device + blockIdx.y];             //z1=zeta_{n-1}^sigma
      ztmp  = (z0 * z1 * (*bp)) /( (*b) * (*a) * (z1 - z0) + z1 * 
				 (*bp)*(1.0 - shift_coeff_d[blockIdx.y] * (*b) ));   // ztmp=zeta_{n+1}^{sigma}

      // bs_double=beta_n^{sigma}
      bs_double = (*b) * ztmp/z0; 

      bs_sh=bs_double;

      css = (*c) * ztmp * ztmp; // since res^{sigma}_{i+1}=zeta^{sigma}_{i+1}*res_{i+1} and *c=(res_{i+1}, res_{i+1})
     
      if(blockIdx.x == 0) 
        {
	//Safely write on global memory
	if (css < residuals_d[blockIdx.y]) converged[blockIdx.y] = 1;

	z_out[ (1-(*iz)) * num_shifts_device + blockIdx.y] = z0;
	z_out[     (*iz) * num_shifts_device + blockIdx.y] = ztmp;	
	bs[blockIdx.y] = bs_double;
        }
      }
    }

  __syncthreads();  //bs_sh and convBlock accessible to all threads

  if(!convBlock) 
    {
    // even sites
    temp_psi1.x = (double) psi[(3*blockIdx.y    )*size_dev + idx].x;
    temp_psi1.y = (double) psi[(3*blockIdx.y    )*size_dev + idx].y;
    temp_psi2.x = (double) psi[(3*blockIdx.y + 1)*size_dev + idx].x;
    temp_psi2.y = (double) psi[(3*blockIdx.y + 1)*size_dev + idx].y;
    temp_psi3.x = (double) psi[(3*blockIdx.y + 2)*size_dev + idx].x;
    temp_psi3.y = (double) psi[(3*blockIdx.y + 2)*size_dev + idx].y;

    temp_psi1.x += (double) psi[(3*blockIdx.y    )*size_dev + idx + f2_offs].x;
    temp_psi1.y += (double) psi[(3*blockIdx.y    )*size_dev + idx + f2_offs].y;
    temp_psi2.x += (double) psi[(3*blockIdx.y + 1)*size_dev + idx + f2_offs].x;
    temp_psi2.y += (double) psi[(3*blockIdx.y + 1)*size_dev + idx + f2_offs].y;
    temp_psi3.x += (double) psi[(3*blockIdx.y + 2)*size_dev + idx + f2_offs].x;
    temp_psi3.y += (double) psi[(3*blockIdx.y + 2)*size_dev + idx + f2_offs].y;
     
    p1 = p[(3*blockIdx.y    )*size_dev + idx];
    p2 = p[(3*blockIdx.y + 1)*size_dev + idx];
    p3 = p[(3*blockIdx.y + 2)*size_dev + idx];

    #ifdef USE_INTRINSIC
    out1.x = __dadd_rn(temp_psi1.x, __dmul_rn(-bs_sh, p1.x));
    out1.y = __dadd_rn(temp_psi1.y, __dmul_rn(-bs_sh, p1.y));
    out2.x = __dadd_rn(temp_psi2.x, __dmul_rn(-bs_sh, p2.x));
    out2.y = __dadd_rn(temp_psi2.y, __dmul_rn(-bs_sh, p2.y));
    out3.x = __dadd_rn(temp_psi3.x, __dmul_rn(-bs_sh, p3.x));
    out3.y = __dadd_rn(temp_psi3.y, __dmul_rn(-bs_sh, p3.y));
    #else
    out1.x = temp_psi1.x - bs_sh * p1.x;
    out1.y = temp_psi1.y - bs_sh * p1.y;
    out2.x = temp_psi2.x - bs_sh * p2.x;
    out2.y = temp_psi2.y - bs_sh * p2.y;
    out3.x = temp_psi3.x - bs_sh * p3.x;
    out3.y = temp_psi3.y - bs_sh * p3.y;
    #endif

    aux_d=out1.x;
    aux_f=(float)aux_d;
    aux_d-=(double)aux_f;
    psi[(3*blockIdx.y    )*size_dev + idx].x = aux_f;
    psi[(3*blockIdx.y    )*size_dev + idx + f2_offs].x = (float) aux_d;

    aux_d=out1.y;
    aux_f=(float)aux_d;
    aux_d-=(double)aux_f;
    psi[(3*blockIdx.y    )*size_dev + idx].y = aux_f;
    psi[(3*blockIdx.y    )*size_dev + idx + f2_offs].y = (float) aux_d;

    aux_d=out2.x;
    aux_f=(float)aux_d;
    aux_d-=(double)aux_f;
    psi[(3*blockIdx.y + 1)*size_dev + idx].x = aux_f;
    psi[(3*blockIdx.y + 1)*size_dev + idx + f2_offs].x = (float) aux_d;

    aux_d=out2.y;
    aux_f=(float)aux_d;
    aux_d-=(double)aux_f;
    psi[(3*blockIdx.y + 1)*size_dev + idx].y = aux_f;
    psi[(3*blockIdx.y + 1)*size_dev + idx + f2_offs].y = (float) aux_d;

    aux_d=out3.x;
    aux_f=(float) aux_d;
    aux_d-= (double) aux_f;
    psi[(3*blockIdx.y + 2)*size_dev + idx].x = aux_f;
    psi[(3*blockIdx.y + 2)*size_dev + idx + f2_offs].x = (float) aux_d;

    aux_d=out3.y;
    aux_f=(float)aux_d;
    aux_d-=(double)aux_f;
    psi[(3*blockIdx.y + 2)*size_dev + idx].y = aux_f;
    psi[(3*blockIdx.y + 2)*size_dev + idx + f2_offs].y = (float) aux_d;
    }
  }



/*
===================================================================== END OF KERNELS
*/



void InitR(double2 *r,
           float2 *in, 
           const int off_2f) 
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside InitR ...\033[0m\n");
  #endif

  dim3 ShiftsBlock(NUM_THREADS);
  dim3 ShiftsGrid(size/(ShiftsBlock.x));

  InitRKernel<<<ShiftsGrid, ShiftsBlock>>>(r, in, off_2f);
  cudaCheckError(AT,"InitRKernel");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated InitR\n");
  #endif
  }



void CalculateNewResidualD(double* prev_res,      // previous residual
			   double* res,           // residual
			   double* b_prev,        // previous beta
			   double* b,             // beta
			   double* d,             // d=<p,A.p>
			   double2* residual_vect, // residual vector
			   double2* A_p)           // A.p
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside CalculateNewResidualD ...\033[0m\n");
  #endif

  // prev_res  = cp  ;  res = c ; residual_vect = r ; A_p = MMp

  unsigned int threads, sharedmem; 
  
  dim3 ResBlock(128);  // must be 128, see CalcResidual
  const unsigned int grid_size_limit = 1 << (int)ceil(log2((double)size/(2.0*(double)ResBlock.x)));  //Number of blocks 
  const unsigned int grid_size = (grid_size_limit < 64) ? grid_size_limit : 64;
  dim3 ResGrid(grid_size);

  double *temp_res;
  cudaSafe(AT,cudaMalloc((void**)&temp_res, sizeof(double)*grid_size), "cudaMalloc");

  cudaSafe(AT,cudaMemcpy(prev_res, res, sizeof(double), cudaMemcpyDeviceToDevice), "cudaMalloc");
  cudaSafe(AT,cudaMemcpy(  b_prev,   b, sizeof(double), cudaMemcpyDeviceToDevice), "cudaMalloc");

  CalculateNewResidualDKernel<<<ResGrid, ResBlock>>>(b, prev_res, d, temp_res, residual_vect, A_p);
  cudaCheckError(AT,"CalculateNewResidualDKernel");
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
  printf("\033[32m\tterminated CalculateNewResidualD\033[0m\n");
  #endif
  }



void InitializeShiftVarsD(double *bs, 
			  double *z, 
			  double *b,
			  double2 *chi_dev, 
			  float2 *psi_dev,
                          int f2_offs, 
			  const int num_shifts)
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside InitializeShiftVarsD ...\033[0m\n");
  #endif

  dim3 ShiftsBlock(NUM_THREADS);
  dim3 ShiftsGrid(size/(2*ShiftsBlock.x), num_shifts);

  InitializeShiftVarsDKernel<<<ShiftsGrid, ShiftsBlock>>>(bs, z, b, chi_dev, psi_dev, f2_offs);
  cudaCheckError(AT,"InitializeShiftVarsDKernel");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated InitializeShiftVarsD\033[0m\n");
  #endif
  }



void UpdatePsD(double *a, 
	       double *c, 
	       double *cp, 
	       double *z, 
	       double *bs, 
	       double *b, 
	       double2 *p_0, 
	       double2 *r, 
	       double2 *p,
	       int *conv,
	       int *iz,
	       const int num_shifts)
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside UpdatePsD ...\033[0m\n");
  #endif

  dim3 UpdatePsBlock(NUM_THREADS);
  dim3 UpdatePsGrid(size/(2*UpdatePsBlock.x), num_shifts);

  UpdatePsDKernel<<<UpdatePsGrid, UpdatePsBlock>>>(a, c, cp, z, bs, b, p_0, r, p, conv, iz);
  cudaCheckError(AT,"UpdatePsDKernel");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated UpdatePsD\033[0m\n");
  #endif
  }



void NewShiftVarsD(double *bs, 
		   double *bp,
		   double *z, 
		   double *b, 
		   double *a, 
		   double *c,
		   float2 *psi_dev,
                   int f2_offs,
	           double2 *p, 
		   int *conv,
		   int *iz,
		   const int num_shifts)
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside NewShiftVarsD ...\033[0m\n");
  #endif

  double *z_out;
  cudaSafe(AT,cudaMalloc((void**)&z_out ,2*num_shifts*sizeof(double)), "cudaSafe");

  dim3 ShiftsBlock(NUM_THREADS);
  dim3 ShiftsGrid(size/(2*ShiftsBlock.x), num_shifts);

  NewShiftVarsDKernel<<<ShiftsGrid, ShiftsBlock>>>(bs, bp, z, z_out, b, a,  c, psi_dev, f2_offs, p, conv, iz);
  cudaCheckError(AT,"NewShiftVarsDKernel");

  cudaSafe(AT,cudaMemcpy(z, z_out, 2*num_shifts*sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy");

  cudaSafe(AT,cudaFree(z_out), "cudaFree");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated NewShiftVarsD\033[0m\n");
  #endif
  }




/*
============================================================ EXTERNAL C FUNCTION
*/



// solve (D^{dag}D+shift)psi_device=mf_device



extern "C" void cuda_shifted_inverter_d(const double residual, 
			                const double *shifts,
			                const int num_shifts,
                                        const int psferm,
			                int *ncount)
  {
  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  printf("DEBUG: inside cuda_shifted_inverter_d  ...\n");
  #endif

  int iter, k, conv;

  double2 *r;                   //residual vector 
  double2 *p_0, *Mp, *MMp, *p;  //auxiliary fermion fields
  double *a, *b, *bp, *bs, *c, *cp, *d, *z;
  int *check_conv, *iz, check_conv_host[max_approx_order];
  double res_sq[num_shifts], shifts_d[num_shifts];

  int iter_iz = 1;

  const int ps_offset=3*size*num_shifts*psferm;
  const int offset_2f=3*size*num_shifts*no_ps;

  size_t vector_size_f = sizeof(float2)*3*size ;  // both even and odd, so 3*size complex number
  size_t vector_size_d = sizeof(double2)*3*size ; // = 2*vector_size_d 

  //Put everything in global memory space at this stage
  cudaSafe(AT,cudaMalloc((void**)&p          ,  num_shifts*vector_size_d), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&r          ,             vector_size_d), "cudaMalloc"); 
  cudaSafe(AT,cudaMalloc((void**)&p_0        ,             vector_size_d), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&Mp         ,             vector_size_d), "cudaMalloc");
  cudaSafe(AT,cudaMalloc((void**)&MMp        ,             vector_size_d), "cudaMalloc");

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
    res_sq[iter] = (residual * residual);
    }

  for(iter = 0; iter < num_shifts; iter++)  
    {
    shifts_d[iter] =  shifts[iter]; 
    }

  // constants
  cudaSafe(AT,cudaMemcpyToSymbol(shift_coeff_d, shifts_d, sizeof(double)*num_shifts, 0, cudaMemcpyHostToDevice), 
                                                                           "cudaMemcpyToSymbol");
  cudaSafe(AT,cudaMemcpyToSymbol(residuals_d, res_sq, sizeof(double)*num_shifts, 0, cudaMemcpyHostToDevice), 
                                                                           "cudaMemcpyToSymbol");
  cudaSafe(AT,cudaMemcpyToSymbol(num_shifts_device, &num_shifts, sizeof(int), 0,   cudaMemcpyHostToDevice), 
                                                                           "cudaMemcpyToSymbol");

  // initialize to 0 smf_device
  // 1st float(s)
  cudaSafe(AT,cudaMemset(smf_device + ps_offset, 0, num_shifts*vector_size_f), "cudaMemset");
  // 2nd float(s)
  cudaSafe(AT,cudaMemset(smf_device + ps_offset + offset_2f, 0, num_shifts*vector_size_f), "cudaMemset"); 

  // initialize check counters
  cudaSafe(AT,cudaMemset(check_conv, 0, num_shifts*sizeof(int)), "cudaMemset");
  cudaSafe(AT,cudaMemcpy(iz, &iter_iz, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");

  // r[0]=chi                   
  InitR(r, mf_device + 3*size*psferm, no_ps*3*size);
                                 //   offset for the 2nd float
  // p[0]=chi
  cudaSafe(AT,cudaMemcpy(p_0, r, vector_size_d, cudaMemcpyDeviceToDevice), "cudaMemcpy");

  // c=||r||^2
  Norm2D(c, r);
  //DebugPrintDouble(c, 0);

  // p[num_shifts] := chi
  for (iter = 0; iter < num_shifts; iter++)
    { 
    cudaSafe(AT,cudaMemcpy(p+3*size*iter, r, vector_size_d, cudaMemcpyDeviceToDevice), "cudaMemcpy");
    }

  //  b[0] := - | r[0] |**2 / < p[0], Ap[0] > ;  A=D^{dag}D
  //  First compute  d  =  < p, A.p > 
  //  Mp=D.p_0
  DslashOperatorDDEO(Mp, p_0, PLUS);

  //  d =  < Mp, Mp > =<p, A.p> 
  Norm2D(d, Mp);

  // MMp = D^{dag}Mp
  DslashOperatorDDEO(MMp, Mp, MINUS);

  //  b = -cp/d
  //  r[1] += b[0] A.p[0] 
  //  c = |r[1]|^2   
  CalculateNewResidualD(cp, c, bp, b, d, r, MMp);

  //  psi[1] -= b[0] p[0]
  InitializeShiftVarsD(bs, z, b, p_0, smf_device + ps_offset, offset_2f, num_shifts);

  conv = 0;
  for (k=1; k < max_cg && !conv ; ++k) 
   {
    //  a[k+1] := |r[k]|**2 / |r[k-1]|**2 ; 
    //  Update p
    //  p[k+1] := r[k+1] + a[k+1] p[k]; 
    //  Compute the shifted as 
    //  ps[k+1] := zs[k+1] r[k+1] + a[k+1] ps[k];
    UpdatePsD(a, c, cp, z, bs, b, p_0, r, p, check_conv, iz, num_shifts);

    // Mp = D p_0
    DslashOperatorDDEO(Mp, p_0, PLUS);

    //  d =  < Mp, Mp > =<p, A.p> 
    Norm2D(d, Mp);

    // MMp = D^{dag} Mp
    DslashOperatorDDEO(MMp, Mp, MINUS);

    CalculateNewResidualD(cp, c, bp, b, d, r, MMp);

    iter_iz = 1-iter_iz; 
    cudaSafe(AT,cudaMemcpy(iz,  &iter_iz, sizeof(int), cudaMemcpyHostToDevice), "cudaSafe");

    NewShiftVarsD(bs, bp, z, b, a, c, smf_device + ps_offset, offset_2f, p, check_conv, iz, num_shifts);

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
  size_t gauge_field_size_f = sizeof(float4)*3*no_links;
  cudaSafe(AT,cudaMemcpy(gauge_field_packed, gauge_field_device, 2*gauge_field_size_f, cudaMemcpyDeviceToHost), 
           "cudaMemcpy");

  // copy back solution to host
  for(k=0; k<num_shifts; k++)
     {                                                         // ps_offset=3*size*num_shifts*psferm
     // 1st float
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k  )*size + ps_offset, 
                            smf_device + (3*k  )*size + ps_offset, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k+1)*size + ps_offset, 
                            smf_device + (3*k+1)*size + ps_offset, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k+2)*size + ps_offset, 
                            smf_device + (3*k+2)*size + ps_offset, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     // 2nd float
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k  )*size + ps_offset + offset_2f, 
                            smf_device + (3*k  )*size + ps_offset + offset_2f, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k+1)*size + ps_offset + offset_2f, 
                            smf_device + (3*k+1)*size + ps_offset + offset_2f, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(psi_packed + (3*k+2)*size + ps_offset + offset_2f, 
                            smf_device + (3*k+2)*size + ps_offset + offset_2f, 
                                                 sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
     }

  // copy back rhs to host
  // 1st float
  cudaSafe(AT,cudaMemcpy(chi_packed + 3*size*psferm, 
                         mf_device + 3*size*psferm,          sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(chi_packed + 3*size*psferm + size, 
                         mf_device + 3*size*psferm + size, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(chi_packed + 3*size*psferm + 2*size, 
                         mf_device + 3*size*psferm + 2*size, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  // 2nd float
  cudaSafe(AT,cudaMemcpy(chi_packed + 3*size*psferm + no_ps*3*size, 
            mf_device + 3*size*psferm          + no_ps*3*size, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(chi_packed + 3*size*psferm + size + no_ps*3*size, 
            mf_device + 3*size*psferm +   size + no_ps*3*size, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
  cudaSafe(AT,cudaMemcpy(chi_packed + 3*size*psferm + 2*size + no_ps*3*size, 
            mf_device + 3*size*psferm + 2*size + no_ps*3*size, sizeh*sizeof(float2), cudaMemcpyDeviceToHost), "cudaMemcpy");
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
  printf("\tterminated cuda_shifted_inverter_d\n");
  #endif
  }

