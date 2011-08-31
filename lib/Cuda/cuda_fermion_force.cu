
__global__ void FermionProductEvenKernel(const float2 *ferm,
                                         const float2 *Mferm,
                                         const int *table,
                                         const int *phases,
                                         int iter,
                                         float2 *aux_l)
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = even <sizeh
  int mu;

  __shared__ int site_table[NUM_THREADS];

  float2 ferm_in_0, ferm_in_1, ferm_in_2;
  float2 Mferm_in_0, Mferm_in_1, Mferm_in_2;

  #ifdef IM_CHEM_POT 
  float2 ferm_aux_0, ferm_aux_1, ferm_aux_2;
  #endif

  ferm_in_0=ferm[             idx];
  ferm_in_1=ferm[  size_dev + idx];
  ferm_in_2=ferm[2*size_dev + idx];

  for(mu=0; mu<3; mu++)
     {
     site_table[threadIdx.x]=table[idx+(4+mu)*size_dev]; // nnp[idx][mu] 

     Mferm_in_0=Mferm[             site_table[threadIdx.x]];
     Mferm_in_1=Mferm[  size_dev + site_table[threadIdx.x]];
     Mferm_in_2=Mferm[2*size_dev + site_table[threadIdx.x]];

     // ris=Mferm^~ferm
     aux_l[idx +              9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_0.x*ferm_in_0.x + Mferm_in_0.y*ferm_in_0.y);
     aux_l[idx +              9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_0.x*ferm_in_0.y + Mferm_in_0.y*ferm_in_0.x);
     aux_l[idx +   size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_0.x*ferm_in_1.x + Mferm_in_0.y*ferm_in_1.y);
     aux_l[idx +   size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_0.x*ferm_in_1.y + Mferm_in_0.y*ferm_in_1.x);
     aux_l[idx + 2*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_0.x*ferm_in_2.x + Mferm_in_0.y*ferm_in_2.y);
     aux_l[idx + 2*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_0.x*ferm_in_2.y + Mferm_in_0.y*ferm_in_2.x);

     aux_l[idx + 3*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_1.x*ferm_in_0.x + Mferm_in_1.y*ferm_in_0.y);
     aux_l[idx + 3*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_1.x*ferm_in_0.y + Mferm_in_1.y*ferm_in_0.x);
     aux_l[idx + 4*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_1.x*ferm_in_1.x + Mferm_in_1.y*ferm_in_1.y);
     aux_l[idx + 4*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_1.x*ferm_in_1.y + Mferm_in_1.y*ferm_in_1.x);
     aux_l[idx + 5*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_1.x*ferm_in_2.x + Mferm_in_1.y*ferm_in_2.y);
     aux_l[idx + 5*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_1.x*ferm_in_2.y + Mferm_in_1.y*ferm_in_2.x);

     aux_l[idx + 6*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_2.x*ferm_in_0.x + Mferm_in_2.y*ferm_in_0.y);
     aux_l[idx + 6*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_2.x*ferm_in_0.y + Mferm_in_2.y*ferm_in_0.x);
     aux_l[idx + 7*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_2.x*ferm_in_1.x + Mferm_in_2.y*ferm_in_1.y);
     aux_l[idx + 7*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_2.x*ferm_in_1.y + Mferm_in_2.y*ferm_in_1.x);
     aux_l[idx + 8*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_2.x*ferm_in_2.x + Mferm_in_2.y*ferm_in_2.y);
     aux_l[idx + 8*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_2.x*ferm_in_2.y + Mferm_in_2.y*ferm_in_2.x);
     }

  for(mu=3; mu<4; mu++)
     {
     site_table[threadIdx.x]=table[idx+(4+mu)*size_dev]; // nnp[idx][mu] 

     #ifndef IM_CHEM_POT
       Mferm_in_0=Mferm[             site_table[threadIdx.x]];
       Mferm_in_1=Mferm[  size_dev + site_table[threadIdx.x]];
       Mferm_in_2=Mferm[2*size_dev + site_table[threadIdx.x]];
     #else
       ferm_aux_0=Mferm[             site_table[threadIdx.x]];
       ferm_aux_1=Mferm[  size_dev + site_table[threadIdx.x]];
       ferm_aux_2=Mferm[2*size_dev + site_table[threadIdx.x]];

       Mferm_in_0.x=ferm_aux_0.x*dev_eim_cos_f-ferm_aux_0.y*dev_eim_sin_f; // Re(ferm_aux_0*e^{imu})
       Mferm_in_0.y=ferm_aux_0.x*dev_eim_sin_f+ferm_aux_0.y*dev_eim_cos_f; // Im(ferm_aux_0*e^{imu})

       Mferm_in_1.x=ferm_aux_1.x*dev_eim_cos_f-ferm_aux_1.y*dev_eim_sin_f; // Re(ferm_aux_1*e^{imu})
       Mferm_in_1.y=ferm_aux_1.x*dev_eim_sin_f+ferm_aux_1.y*dev_eim_cos_f; // Im(ferm_aux_1*e^{imu})

       Mferm_in_2.x=ferm_aux_2.x*dev_eim_cos_f-ferm_aux_2.y*dev_eim_sin_f; // Re(ferm_aux_2*e^{imu})
       Mferm_in_2.y=ferm_aux_2.x*dev_eim_sin_f+ferm_aux_2.y*dev_eim_cos_f; // Im(ferm_aux_2*e^{imu})
     #endif

     // ris=Mferm^~ferm
     aux_l[idx +              9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_0.x*ferm_in_0.x + Mferm_in_0.y*ferm_in_0.y);
     aux_l[idx +              9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_0.x*ferm_in_0.y + Mferm_in_0.y*ferm_in_0.x);
     aux_l[idx +   size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_0.x*ferm_in_1.x + Mferm_in_0.y*ferm_in_1.y);
     aux_l[idx +   size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_0.x*ferm_in_1.y + Mferm_in_0.y*ferm_in_1.x);
     aux_l[idx + 2*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_0.x*ferm_in_2.x + Mferm_in_0.y*ferm_in_2.y);
     aux_l[idx + 2*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_0.x*ferm_in_2.y + Mferm_in_0.y*ferm_in_2.x);

     aux_l[idx + 3*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_1.x*ferm_in_0.x + Mferm_in_1.y*ferm_in_0.y);
     aux_l[idx + 3*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_1.x*ferm_in_0.y + Mferm_in_1.y*ferm_in_0.x);
     aux_l[idx + 4*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_1.x*ferm_in_1.x + Mferm_in_1.y*ferm_in_1.y);
     aux_l[idx + 4*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_1.x*ferm_in_1.y + Mferm_in_1.y*ferm_in_1.x);
     aux_l[idx + 5*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_1.x*ferm_in_2.x + Mferm_in_1.y*ferm_in_2.y);
     aux_l[idx + 5*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_1.x*ferm_in_2.y + Mferm_in_1.y*ferm_in_2.x);

     aux_l[idx + 6*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_2.x*ferm_in_0.x + Mferm_in_2.y*ferm_in_0.y);
     aux_l[idx + 6*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_2.x*ferm_in_0.y + Mferm_in_2.y*ferm_in_0.x);
     aux_l[idx + 7*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_2.x*ferm_in_1.x + Mferm_in_2.y*ferm_in_1.y);
     aux_l[idx + 7*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_2.x*ferm_in_1.y + Mferm_in_2.y*ferm_in_1.x);
     aux_l[idx + 8*size_dev + 9*mu*size_dev].x+=shift_coeff[iter]*( Mferm_in_2.x*ferm_in_2.x + Mferm_in_2.y*ferm_in_2.y);
     aux_l[idx + 8*size_dev + 9*mu*size_dev].y+=shift_coeff[iter]*(-Mferm_in_2.x*ferm_in_2.y + Mferm_in_2.y*ferm_in_2.x);
     }
  }






__global__ void FermionProductOddKernel(const float2 *ferm,
                                        const float2 *Mferm,
                                        const int *table,
                                        const int *phases,
                                        int iter,
                                        float2 *aux_l)
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x+size_dev/2;   // idx = odd >sizeh
  int mu;

  __shared__ int site_table[NUM_THREADS];

  float2 ferm_in_0, ferm_in_1, ferm_in_2;
  float2 Mferm_in_0, Mferm_in_1, Mferm_in_2;

  #ifdef IM_CHEM_POT 
  float2 ferm_aux_0, ferm_aux_1, ferm_aux_2;
  #endif

  Mferm_in_0=Mferm[             idx];
  Mferm_in_1=Mferm[  size_dev + idx];
  Mferm_in_2=Mferm[2*size_dev + idx];

  for(mu=0; mu<3; mu++)
     {
     site_table[threadIdx.x]=table[idx +(4+mu)*size_dev]; // nnp[idx][mu] 

     ferm_in_0=ferm[             site_table[threadIdx.x]];
     ferm_in_1=ferm[  size_dev + site_table[threadIdx.x]];
     ferm_in_2=ferm[2*size_dev + site_table[threadIdx.x]];

     // ris=ferm^~Mferm
     aux_l[idx +              9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_0.x*Mferm_in_0.x + ferm_in_0.y*Mferm_in_0.y);
     aux_l[idx +              9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_0.x*Mferm_in_0.y + ferm_in_0.y*Mferm_in_0.x);
     aux_l[idx + 1*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_0.x*Mferm_in_1.x + ferm_in_0.y*Mferm_in_1.y);
     aux_l[idx + 1*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_0.x*Mferm_in_1.y + ferm_in_0.y*Mferm_in_1.x);
     aux_l[idx + 2*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_0.x*Mferm_in_2.x + ferm_in_0.y*Mferm_in_2.y);
     aux_l[idx + 2*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_0.x*Mferm_in_2.y + ferm_in_0.y*Mferm_in_2.x);

     aux_l[idx + 3*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_1.x*Mferm_in_0.x + ferm_in_1.y*Mferm_in_0.y);
     aux_l[idx + 3*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_1.x*Mferm_in_0.y + ferm_in_1.y*Mferm_in_0.x);
     aux_l[idx + 4*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_1.x*Mferm_in_1.x + ferm_in_1.y*Mferm_in_1.y);
     aux_l[idx + 4*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_1.x*Mferm_in_1.y + ferm_in_1.y*Mferm_in_1.x);
     aux_l[idx + 5*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_1.x*Mferm_in_2.x + ferm_in_1.y*Mferm_in_2.y);
     aux_l[idx + 5*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_1.x*Mferm_in_2.y + ferm_in_1.y*Mferm_in_2.x);

     aux_l[idx + 6*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_2.x*Mferm_in_0.x + ferm_in_2.y*Mferm_in_0.y);
     aux_l[idx + 6*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_2.x*Mferm_in_0.y + ferm_in_2.y*Mferm_in_0.x);
     aux_l[idx + 7*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_2.x*Mferm_in_1.x + ferm_in_2.y*Mferm_in_1.y);
     aux_l[idx + 7*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_2.x*Mferm_in_1.y + ferm_in_2.y*Mferm_in_1.x);
     aux_l[idx + 8*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_2.x*Mferm_in_2.x + ferm_in_2.y*Mferm_in_2.y);
     aux_l[idx + 8*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_2.x*Mferm_in_2.y + ferm_in_2.y*Mferm_in_2.x);
    }
  for(mu=3; mu<4; mu++)
     {
     site_table[threadIdx.x]=table[idx +(4+mu)*size_dev]; // nnp[idx][mu] 

     #ifndef IM_CHEM_POT
       ferm_in_0=ferm[             site_table[threadIdx.x]];
       ferm_in_1=ferm[  size_dev + site_table[threadIdx.x]];
       ferm_in_2=ferm[2*size_dev + site_table[threadIdx.x]];
     #else
       ferm_aux_0=ferm[             site_table[threadIdx.x]];
       ferm_aux_1=ferm[  size_dev + site_table[threadIdx.x]];
       ferm_aux_2=ferm[2*size_dev + site_table[threadIdx.x]];

       ferm_in_0.x=ferm_aux_0.x*dev_eim_cos_f-ferm_aux_0.y*dev_eim_sin_f; // Re(ferm_aux_0*e^{imu})
       ferm_in_0.y=ferm_aux_0.x*dev_eim_sin_f+ferm_aux_0.y*dev_eim_cos_f; // Im(ferm_aux_0*e^{imu})

       ferm_in_1.x=ferm_aux_1.x*dev_eim_cos_f-ferm_aux_1.y*dev_eim_sin_f; // Re(ferm_aux_1*e^{imu})
       ferm_in_1.y=ferm_aux_1.x*dev_eim_sin_f+ferm_aux_1.y*dev_eim_cos_f; // Im(ferm_aux_1*e^{imu})

       ferm_in_2.x=ferm_aux_2.x*dev_eim_cos_f-ferm_aux_2.y*dev_eim_sin_f; // Re(ferm_aux_2*e^{imu})
       ferm_in_2.y=ferm_aux_2.x*dev_eim_sin_f+ferm_aux_2.y*dev_eim_cos_f; // Im(ferm_aux_2*e^{imu})
     #endif

     // ris=ferm^~Mferm
     aux_l[idx +              9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_0.x*Mferm_in_0.x + ferm_in_0.y*Mferm_in_0.y);
     aux_l[idx +              9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_0.x*Mferm_in_0.y + ferm_in_0.y*Mferm_in_0.x);
     aux_l[idx + 1*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_0.x*Mferm_in_1.x + ferm_in_0.y*Mferm_in_1.y);
     aux_l[idx + 1*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_0.x*Mferm_in_1.y + ferm_in_0.y*Mferm_in_1.x);
     aux_l[idx + 2*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_0.x*Mferm_in_2.x + ferm_in_0.y*Mferm_in_2.y);
     aux_l[idx + 2*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_0.x*Mferm_in_2.y + ferm_in_0.y*Mferm_in_2.x);

     aux_l[idx + 3*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_1.x*Mferm_in_0.x + ferm_in_1.y*Mferm_in_0.y);
     aux_l[idx + 3*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_1.x*Mferm_in_0.y + ferm_in_1.y*Mferm_in_0.x);
     aux_l[idx + 4*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_1.x*Mferm_in_1.x + ferm_in_1.y*Mferm_in_1.y);
     aux_l[idx + 4*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_1.x*Mferm_in_1.y + ferm_in_1.y*Mferm_in_1.x);
     aux_l[idx + 5*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_1.x*Mferm_in_2.x + ferm_in_1.y*Mferm_in_2.y);
     aux_l[idx + 5*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_1.x*Mferm_in_2.y + ferm_in_1.y*Mferm_in_2.x);

     aux_l[idx + 6*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_2.x*Mferm_in_0.x + ferm_in_2.y*Mferm_in_0.y);
     aux_l[idx + 6*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_2.x*Mferm_in_0.y + ferm_in_2.y*Mferm_in_0.x);
     aux_l[idx + 7*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_2.x*Mferm_in_1.x + ferm_in_2.y*Mferm_in_1.y);
     aux_l[idx + 7*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_2.x*Mferm_in_1.y + ferm_in_2.y*Mferm_in_1.x);
     aux_l[idx + 8*size_dev + 9*mu*size_dev].x-=shift_coeff[iter]*( ferm_in_2.x*Mferm_in_2.x + ferm_in_2.y*Mferm_in_2.y);
     aux_l[idx + 8*size_dev + 9*mu*size_dev].y-=shift_coeff[iter]*(-ferm_in_2.x*Mferm_in_2.y + ferm_in_2.y*Mferm_in_2.x);
    }
  }



__global__ void UmulTAKernel(int *phases,
                             float2 *aux_l,
                             float4 *ipdot_l,
                             size_t gauge_offset)
  {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;   // 0<= idx < size
  int stag_phase;

  DeclareMatrixRegs;

  float2 matrix_00, matrix_01, matrix_02,
         matrix_10, matrix_11, matrix_12,
         matrix_20, matrix_21, matrix_22;
  
  float2 mat0, mat1, mat2; 

  matrix_00=aux_l[idx              + 9*blockIdx.y*size_dev];
  matrix_01=aux_l[idx + 1*size_dev + 9*blockIdx.y*size_dev];
  matrix_02=aux_l[idx + 2*size_dev + 9*blockIdx.y*size_dev];

  matrix_10=aux_l[idx + 3*size_dev + 9*blockIdx.y*size_dev];
  matrix_11=aux_l[idx + 4*size_dev + 9*blockIdx.y*size_dev];
  matrix_12=aux_l[idx + 5*size_dev + 9*blockIdx.y*size_dev];

  matrix_20=aux_l[idx + 6*size_dev + 9*blockIdx.y*size_dev];
  matrix_21=aux_l[idx + 7*size_dev + 9*blockIdx.y*size_dev];
  matrix_22=aux_l[idx + 8*size_dev + 9*blockIdx.y*size_dev];

  stag_phase=phases[idx+blockIdx.y*size_dev];

  LoadLinkRegs(gauge_texRef, size_dev, idx + gauge_offset, blockIdx.y); 

  mat0.x = link0.x*matrix_00.x - link0.y*matrix_00.y +
           link0.z*matrix_10.x - link0.w*matrix_10.y +
           link1.x*matrix_20.x - link1.y*matrix_20.y;

  mat0.y = link0.x*matrix_00.y + link0.y*matrix_00.x +
           link0.z*matrix_10.y + link0.w*matrix_10.x +
           link1.x*matrix_20.y + link1.y*matrix_20.x;

  mat1.x = link1.z*matrix_00.x - link1.w*matrix_00.y +
           link2.x*matrix_10.x - link2.y*matrix_10.y +
           link2.z*matrix_20.x - link2.w*matrix_20.y;

  mat1.y = link1.z*matrix_00.y + link1.w*matrix_00.x +
           link2.x*matrix_10.y + link2.y*matrix_10.x +
           link2.z*matrix_20.y + link2.w*matrix_20.x;
 
  mat2.x = stag_phase*(C1RE*matrix_00.x - C1IM*matrix_00.y +
                       C2RE*matrix_10.x - C2IM*matrix_10.y +
                       C3RE*matrix_20.x - C3IM*matrix_20.y);

  mat2.y = stag_phase*(C1RE*matrix_00.y + C1IM*matrix_00.x +
                       C2RE*matrix_10.y + C2IM*matrix_10.x +
                       C3RE*matrix_20.y + C3IM*matrix_20.x);

  matrix_00 = mat0;
  matrix_10 = mat1;
  matrix_20 = mat2;
 
  mat0.x = link0.x*matrix_01.x - link0.y*matrix_01.y +
           link0.z*matrix_11.x - link0.w*matrix_11.y +
           link1.x*matrix_21.x - link1.y*matrix_21.y;

  mat0.y = link0.x*matrix_01.y + link0.y*matrix_01.x +
           link0.z*matrix_11.y + link0.w*matrix_11.x +
           link1.x*matrix_21.y + link1.y*matrix_21.x;

  mat1.x = link1.z*matrix_01.x - link1.w*matrix_01.y +
           link2.x*matrix_11.x - link2.y*matrix_11.y +
           link2.z*matrix_21.x - link2.w*matrix_21.y;

  mat1.y = link1.z*matrix_01.y + link1.w*matrix_01.x +
           link2.x*matrix_11.y + link2.y*matrix_11.x +
           link2.z*matrix_21.y + link2.w*matrix_21.x;

  mat2.x = stag_phase*(C1RE*matrix_01.x - C1IM*matrix_01.y +
                       C2RE*matrix_11.x - C2IM*matrix_11.y +
                       C3RE*matrix_21.x - C3IM*matrix_21.y);

  mat2.y = stag_phase*(C1RE*matrix_01.y + C1IM*matrix_01.x +
                       C2RE*matrix_11.y + C2IM*matrix_11.x +
                       C3RE*matrix_21.y + C3IM*matrix_21.x);

  matrix_01 = mat0;
  matrix_11 = mat1;
  matrix_21 = mat2;

  mat0.x = link0.x*matrix_02.x - link0.y*matrix_02.y +
           link0.z*matrix_12.x - link0.w*matrix_12.y +
           link1.x*matrix_22.x - link1.y*matrix_22.y;

  mat0.y = link0.x*matrix_02.y + link0.y*matrix_02.x +
           link0.z*matrix_12.y + link0.w*matrix_12.x +
           link1.x*matrix_22.y + link1.y*matrix_22.x;

  mat1.x = link1.z*matrix_02.x - link1.w*matrix_02.y +
           link2.x*matrix_12.x - link2.y*matrix_12.y +
           link2.z*matrix_22.x - link2.w*matrix_22.y;

  mat1.y = link1.z*matrix_02.y + link1.w*matrix_02.x +
           link2.x*matrix_12.y + link2.y*matrix_12.x +
           link2.z*matrix_22.y + link2.w*matrix_22.x;

  mat2.x = stag_phase*(C1RE*matrix_02.x - C1IM*matrix_02.y +
                       C2RE*matrix_12.x - C2IM*matrix_12.y +
                       C3RE*matrix_22.x - C3IM*matrix_22.y);

  mat2.y = stag_phase*(C1RE*matrix_02.y + C1IM*matrix_02.x +
                       C2RE*matrix_12.y + C2IM*matrix_12.x +
                       C3RE*matrix_22.y + C3IM*matrix_22.x);

  matrix_02 = mat0;
  matrix_12 = mat1;
  matrix_22 = mat2;

  ipdot_l[idx +            2*blockIdx.y*size_dev].x=0.5f*(matrix_01.x-matrix_10.x);
  ipdot_l[idx +            2*blockIdx.y*size_dev].y=0.5f*(matrix_01.y+matrix_10.y);
  ipdot_l[idx +            2*blockIdx.y*size_dev].z=0.5f*(matrix_02.x-matrix_20.x);
  ipdot_l[idx +            2*blockIdx.y*size_dev].w=0.5f*(matrix_02.y+matrix_20.y);

  ipdot_l[idx + size_dev + 2*blockIdx.y*size_dev].x=0.5f*(matrix_12.x-matrix_21.x);
  ipdot_l[idx + size_dev + 2*blockIdx.y*size_dev].y=0.5f*(matrix_12.y+matrix_21.y);
  ipdot_l[idx + size_dev + 2*blockIdx.y*size_dev].z=matrix_00.y-0.3333333f*(matrix_00.y+matrix_11.y+matrix_22.y);
  ipdot_l[idx + size_dev + 2*blockIdx.y*size_dev].w=matrix_11.y-0.3333333f*(matrix_00.y+matrix_11.y+matrix_22.y);
  }



/*
============================================================================== EXTERNAL C FUNCTION
*/




extern "C" void cuda_fermion_force(int num_shifts, 
                            float *numerators)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_fermion_force ...\n");
  #endif

  int ps, iter;
  int ps_offset;
  float2 *Mf, *aux_dev;

  size_t vector_size   = sizeof(float2)*3*size;
  size_t gauge_field_size = sizeof(float)*no_links*12;
 
  cudaSafe(AT,cudaMalloc((void**)&Mf, vector_size), "cudaMalloc"); 

  cudaSafe(AT,cudaMalloc((void**)&aux_dev, 18*no_links*sizeof(float)), "cudaMalloc"); 
  cudaSafe(AT,cudaMemset(aux_dev, 0, 18*no_links*sizeof(float)), "cudaMemset");

  cudaSafe(AT,cudaMemcpyToSymbol(shift_coeff, numerators, sizeof(float)*num_shifts, 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");

  dim3 BlockDim(NUM_THREADS);
  dim3 GridDim(size/(2*BlockDim.x));

  for(ps=0; ps<no_ps; ps++)
     {
     ps_offset=3*size*ps;

     for(iter=0; iter<num_shifts; iter++)
        {
        CuDoe(Mf, smf_device+num_shifts*ps_offset+3*iter*size);

        FermionProductEvenKernel<<<GridDim, BlockDim>>>(smf_device+num_shifts*ps_offset+3*iter*size, 
                                                  Mf, device_table, device_phases, iter, aux_dev);
        cudaCheckError(AT,"FermionProductEvenKernel"); 

        FermionProductOddKernel<<<GridDim, BlockDim>>>(smf_device+num_shifts*ps_offset+3*iter*size, 
                                                  Mf, device_table, device_phases, iter, aux_dev);
        cudaCheckError(AT,"FermionProductOddKernel"); 
        }
     }

  dim3 BlockDimBis(NUM_THREADS);
  dim3 GridDimBis(size/BlockDimBis.x, 4);

  size_t offset_g;
  cudaSafe(AT,cudaBindTexture(&offset_g, gauge_texRef, gauge_field_device, gauge_field_size), "cudaBindTexture");
  offset_g/=sizeof(float4);

  UmulTAKernel<<<GridDimBis, BlockDimBis>>>(device_phases, aux_dev, ipdot_device, offset_g);
  cudaCheckError(AT,"UmulTAKernel"); 

  cudaSafe(AT,cudaUnbindTexture(gauge_texRef), "cudaUnbindTexture");

  #ifdef PARAMETER_TEST
    double *norm2_dev, norm2_host;
    cudaSafe(AT,cudaMalloc((void**)&norm2_dev, sizeof(double)), "cudaMalloc"); 

    IpdotNorm2(norm2_dev, ipdot_device);
 
    cudaSafe(AT,cudaMemcpy(&norm2_host, norm2_dev, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy"); 

    printf("L2 norm of FERMION force = %g\n", sqrt(norm2_host));
    cudaSafe(AT,cudaFree(norm2_dev), "cudaFree");
  #endif

  cudaSafe(AT,cudaFree(aux_dev), "cudaFree");
  cudaSafe(AT,cudaFree(Mf), "cudaFree");

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_fermion_force\n");
  #endif
  }
