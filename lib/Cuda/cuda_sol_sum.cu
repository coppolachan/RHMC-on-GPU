
__global__ void ChiMultiplyKernel(float2 *chi,
                                  int offs_2f) 
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = even <sizeh

  float aux;
  double2 ferm_in_0, ferm_in_1, ferm_in_2;

  ferm_in_0.x = (double) chi[             idx].x;
  ferm_in_0.y = (double) chi[             idx].y;
  ferm_in_1.x = (double) chi[  size_dev + idx].x;
  ferm_in_1.y = (double) chi[  size_dev + idx].y;
  ferm_in_2.x = (double) chi[2*size_dev + idx].x;
  ferm_in_2.y = (double) chi[2*size_dev + idx].y;

  ferm_in_0.x += (double) chi[             idx + offs_2f].x;
  ferm_in_0.y += (double) chi[             idx + offs_2f].y;
  ferm_in_1.x += (double) chi[  size_dev + idx + offs_2f].x;
  ferm_in_1.y += (double) chi[  size_dev + idx + offs_2f].y;
  ferm_in_2.x += (double) chi[2*size_dev + idx + offs_2f].x;
  ferm_in_2.y += (double) chi[2*size_dev + idx + offs_2f].y;

  ferm_in_0.x*=d_aux_dev;
  ferm_in_0.y*=d_aux_dev;
  ferm_in_1.x*=d_aux_dev;
  ferm_in_1.y*=d_aux_dev;
  ferm_in_2.x*=d_aux_dev;
  ferm_in_2.y*=d_aux_dev;

  aux = (float) ferm_in_0.x;
  chi[             idx].x = aux;
  ferm_in_0.x -= (double) aux;
  chi[             idx + offs_2f].x = (float) ferm_in_0.x;
  
  aux = (float) ferm_in_0.y;
  chi[             idx].y = aux;
  ferm_in_0.y -= (double) aux;
  chi[             idx + offs_2f].y = (float) ferm_in_0.y;

  aux = (float) ferm_in_1.x;
  chi[  size_dev + idx].x = aux;
  ferm_in_1.x -= (double) aux;
  chi[  size_dev + idx + offs_2f].x = (float) ferm_in_1.x;

  aux = (float) ferm_in_1.y;
  chi[  size_dev + idx].y = aux;
  ferm_in_1.y -= (double) aux;
  chi[  size_dev + idx + offs_2f].y = (float) ferm_in_1.y;

  aux = (float) ferm_in_2.x;
  chi[2*size_dev + idx].x = aux;
  ferm_in_2.x -= (double) aux;
  chi[2*size_dev + idx + offs_2f].x = (float) ferm_in_2.x;

  aux = (float) ferm_in_2.y;
  chi[2*size_dev + idx].y = aux;
  ferm_in_2.y -= (double) aux;
  chi[2*size_dev + idx + offs_2f].y = (float) ferm_in_2.y;
  }



__global__ void ChiConstructKernel(float2 *chiferm,
                                   int chi_off_2f,
                                   const float2 *shiftferm,
                                   const int sh_off_2f,
                                   int num_shifts)
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // idx = even <sizeh
  int iter;

  float aux;
  double2 ferm_in_0, ferm_in_1, ferm_in_2;
  double2 aux_0, aux_1, aux_2;

  // 1st float
  ferm_in_0.x = (double) chiferm[             idx].x;
  ferm_in_0.y = (double) chiferm[             idx].y;
  ferm_in_1.x = (double) chiferm[  size_dev + idx].x;
  ferm_in_1.y = (double) chiferm[  size_dev + idx].y;
  ferm_in_2.x = (double) chiferm[2*size_dev + idx].x;
  ferm_in_2.y = (double) chiferm[2*size_dev + idx].y;

  // 2nd float
  ferm_in_0.x += (double) chiferm[             idx + chi_off_2f].x;
  ferm_in_0.y += (double) chiferm[             idx + chi_off_2f].y;
  ferm_in_1.x += (double) chiferm[  size_dev + idx + chi_off_2f].x;
  ferm_in_1.y += (double) chiferm[  size_dev + idx + chi_off_2f].y;
  ferm_in_2.x += (double) chiferm[2*size_dev + idx + chi_off_2f].x;
  ferm_in_2.y += (double) chiferm[2*size_dev + idx + chi_off_2f].y;

  for(iter=0; iter<num_shifts; iter++)
     {
     // 1st float
     aux_0.x = (double) shiftferm[             idx + 3*iter*size_dev].x;
     aux_0.y = (double) shiftferm[             idx + 3*iter*size_dev].y;
     aux_1.x = (double) shiftferm[  size_dev + idx + 3*iter*size_dev].x;
     aux_1.y = (double) shiftferm[  size_dev + idx + 3*iter*size_dev].y;
     aux_2.x = (double) shiftferm[2*size_dev + idx + 3*iter*size_dev].x;
     aux_2.y = (double) shiftferm[2*size_dev + idx + 3*iter*size_dev].y;

     // 2nd float
     aux_0.x += (double) shiftferm[             idx + 3*iter*size_dev + sh_off_2f].x;
     aux_0.y += (double) shiftferm[             idx + 3*iter*size_dev + sh_off_2f].y;
     aux_1.x += (double) shiftferm[  size_dev + idx + 3*iter*size_dev + sh_off_2f].x;
     aux_1.y += (double) shiftferm[  size_dev + idx + 3*iter*size_dev + sh_off_2f].y;
     aux_2.x += (double) shiftferm[2*size_dev + idx + 3*iter*size_dev + sh_off_2f].x;
     aux_2.y += (double) shiftferm[2*size_dev + idx + 3*iter*size_dev + sh_off_2f].y;

     aux_0.x*=shift_coeff_d[iter];
     aux_0.y*=shift_coeff_d[iter];
     aux_1.x*=shift_coeff_d[iter];
     aux_1.y*=shift_coeff_d[iter];
     aux_2.x*=shift_coeff_d[iter];
     aux_2.y*=shift_coeff_d[iter];

     ferm_in_0.x+=aux_0.x;
     ferm_in_0.y+=aux_0.y;
     ferm_in_1.x+=aux_1.x;
     ferm_in_1.y+=aux_1.y;
     ferm_in_2.x+=aux_2.x;
     ferm_in_2.y+=aux_2.y;
     }

  aux = (float) ferm_in_0.x;
  chiferm[             idx].x = aux;
  ferm_in_0.x -= (double) aux;
  chiferm[             idx + chi_off_2f].x = (float) ferm_in_0.x;
  
  aux = (float) ferm_in_0.y;
  chiferm[             idx].y = aux;
  ferm_in_0.y -= (double) aux;
  chiferm[             idx + chi_off_2f].y = (float) ferm_in_0.y;

  aux = (float) ferm_in_1.x;
  chiferm[  size_dev + idx].x = aux;
  ferm_in_1.x -= (double) aux;
  chiferm[  size_dev + idx + chi_off_2f].x = (float) ferm_in_1.x;

  aux = (float) ferm_in_1.y;
  chiferm[  size_dev + idx].y = aux;
  ferm_in_1.y -= (double) aux;
  chiferm[  size_dev + idx + chi_off_2f].y = (float) ferm_in_1.y;

  aux = (float) ferm_in_2.x;
  chiferm[2*size_dev + idx].x = aux;
  ferm_in_2.x -= (double) aux;
  chiferm[2*size_dev + idx + chi_off_2f].x = (float) ferm_in_2.x;

  aux = (float) ferm_in_2.y;
  chiferm[2*size_dev + idx].y = aux;
  ferm_in_2.y -= (double) aux;
  chiferm[2*size_dev + idx + chi_off_2f].y = (float) ferm_in_2.y;
  }



/*
================================================================== EXTERNAL C FUNCTION
*/

//
// Routine to resum the shifted multifermion output of shift-inverter
//


extern "C" void cuda_sol_sum(const int num_shifts, 
                             const double mult_const, 
                             const double *numerators)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_sol_sum ...\n");
  #endif

  int ps, ps_offset;

  const int chi_offs_2f=3*size*no_ps;
  const int  sh_offs_2f=3*size*no_ps*num_shifts;

  dim3 BlockDim(NUM_THREADS);
  dim3 GridDim(size/(2*BlockDim.x));

  double aux=mult_const;
  // set constants
  cudaSafe(AT,cudaMemcpyToSymbol(d_aux_dev, &aux, sizeof(double), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");
  cudaSafe(AT,cudaMemcpyToSymbol(shift_coeff_d, numerators, sizeof(double)*num_shifts, 0, cudaMemcpyHostToDevice), 
         "cudaMemcpyToSymbol");

  for(ps=0; ps<no_ps; ps++)
     {
     ps_offset=3*size*ps;

     ChiMultiplyKernel<<<GridDim, BlockDim>>>(mf_device + ps_offset, chi_offs_2f);
     cudaCheckError(AT,"ChiMultiplyKernel"); 

     ChiConstructKernel<<<GridDim, BlockDim>>>(mf_device + ps_offset, chi_offs_2f, smf_device + num_shifts*ps_offset, 
                               sh_offs_2f, num_shifts);
     cudaCheckError(AT,"ChiConstructKernel"); 

     // copy chi back on cpu
     // 1st float 
     cudaSafe(AT,cudaMemcpy(chi_packed          + ps_offset, 
                                 mf_device          + ps_offset, size*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(chi_packed +   size + ps_offset, 
                                 mf_device +   size + ps_offset, size*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(chi_packed + 2*size + ps_offset, 
                                 mf_device + 2*size + ps_offset, size*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
     // 2nd float
     cudaSafe(AT,cudaMemcpy(chi_packed          + ps_offset + no_ps*3*size, 
                 mf_device          + ps_offset + no_ps*3*size, size*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(chi_packed +   size + ps_offset + no_ps*3*size, 
                 mf_device +   size + ps_offset + no_ps*3*size, size*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
     cudaSafe(AT,cudaMemcpy(chi_packed + 2*size + ps_offset + no_ps*3*size, 
                 mf_device+ 2*size + ps_offset + no_ps*3*size, size*sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
     }

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_sol_sum\n");
  #endif
  }
