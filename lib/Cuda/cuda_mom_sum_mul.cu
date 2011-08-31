
__global__ void MomentaSumMulKernel(float4 *momenta, 
                                    float4 *ipdot)
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
  float4 aux1, aux2, aux3, aux4;
                                                   //     x      y      z     w
  aux1 = momenta[idx + size_dev*(0+2*blockIdx.y)]; // =(re_01, im_01, re_02, im_02)	
  aux2 = momenta[idx + size_dev*(1+2*blockIdx.y)]; // =(re_12, im_12, re_00, re_11)

  aux3 = ipdot[idx + size_dev*(0+2*blockIdx.y)];   // =(re_01, im_01, re_02, im_02) 
  aux4 = ipdot[idx + size_dev*(1+2*blockIdx.y)];   // =(re_12, im_12, im_00, im_11)

  // mom_00 += i*f_aux_dev*ipdot_00
  aux2.z -= f_aux_dev*aux4.z;

  // mom_01 += i*f_aux_dev*ipdot_01
  aux1.x -= f_aux_dev*aux3.y;
  aux1.y += f_aux_dev*aux3.x;

  // mom_02 += i*f_aux_dev*ipdot_01
  aux1.z -= f_aux_dev*aux3.w;
  aux1.w += f_aux_dev*aux3.z;

  // mom_11 += i*f_aux_dev*ipdot_11
  aux2.w -= f_aux_dev*aux4.w;

  // mom_12 += i*f_aux_dev*ipdot_12
  aux2.x -= f_aux_dev*aux4.y;
  aux2.y += f_aux_dev*aux4.x;

  momenta[idx + size_dev*(0+2*blockIdx.y)] = aux1;
  momenta[idx + size_dev*(1+2*blockIdx.y)] = aux2;
  }



/*
================================================================== EXTERNAL C FUNCTION
*/



extern "C" void cuda_momenta_sum_multiply(float step)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_momenta_sum_multiply ...\n");
  #endif

  cudaSafe(AT,cudaMemcpyToSymbol(f_aux_dev, &step, sizeof(float),  0,   cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");

  //Setting block and grid sizes for the kernel
  dim3 BlockDimension(NUM_THREADS);
  dim3 GridDimension(size/BlockDimension.x, 4); //Run separately the four dimensions

  // momenta[i]+=i*p*ipdot_loc[i]   
  MomentaSumMulKernel<<<GridDimension,BlockDimension>>>(momenta_device, ipdot_device); 
  cudaCheckError(AT,"MomentaSumMulKernel");

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_momenta_sum_multiply\n");
  #endif
  }
