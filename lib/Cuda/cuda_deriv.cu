__global__ void DeriveFieldKernel(float4 *out_matrix,
			          const int *table,
                                  const int *phases,
                                  size_t gauge_offset) 
  {
  //Linear index 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int nu, direction;
  int temp_site;
  int stag_phase;

  //New table indexing (index fastest)
  __shared__ int site_table[NUM_THREADS];

  // Staples  
  __shared__ float2 staples[9][NUM_THREADS];

  //Temporary row-column store
  float2 mat0, mat1, mat2;  //6 registers

  DeclareMatrixRegs;  //12 registers

  float2 matrix_00, matrix_01, matrix_02, 
         matrix_10, matrix_11, matrix_12,
         matrix_20, matrix_21, matrix_22; //18 registers


  staples[0][threadIdx.x].x = 0.0f;
  staples[0][threadIdx.x].y = 0.0f;
  staples[1][threadIdx.x].x = 0.0f;
  staples[1][threadIdx.x].y = 0.0f;
  staples[2][threadIdx.x].x = 0.0f;
  staples[2][threadIdx.x].y = 0.0f;

  staples[3][threadIdx.x].x = 0.0f;
  staples[3][threadIdx.x].y = 0.0f;
  staples[4][threadIdx.x].x = 0.0f;
  staples[4][threadIdx.x].y = 0.0f;
  staples[5][threadIdx.x].x = 0.0f;
  staples[5][threadIdx.x].y = 0.0f;

  staples[6][threadIdx.x].x = 0.0f;
  staples[6][threadIdx.x].y = 0.0f;
  staples[7][threadIdx.x].x = 0.0f;
  staples[7][threadIdx.x].y = 0.0f;
  staples[8][threadIdx.x].x = 0.0f;
  staples[8][threadIdx.x].y = 0.0f;
 
  // mu = blockIdx.y
  for(direction = 1; direction < 4; direction++) 
     {
     nu = (direction+blockIdx.y) & 3;  //     nu = (direction+blockIdx.y) % 4;

//             (1)
//           +-->--+
//           |     |
// ^mu       |     V  (2)   to calculate (1)*(2)*(3)
// |         |     |
//           +--<--+
// ->nu   idx  (3)

     site_table[threadIdx.x] = table[idx+(4+blockIdx.y)*size_dev];            // idx+mu
     LoadLinkRegs(gauge_texRef, size_dev, site_table[threadIdx.x] + gauge_offset, nu);  // U(idx+mu)_nu  
     stag_phase=phases[site_table[threadIdx.x]+nu*size_dev];

     matrix_00.x = f_aux_dev*link0.x;
     matrix_00.y = f_aux_dev*link0.y;

     matrix_01.x = f_aux_dev*link0.z;
     matrix_01.y = f_aux_dev*link0.w;

     matrix_02.x = f_aux_dev*link1.x;
     matrix_02.y = f_aux_dev*link1.y;

     matrix_10.x = f_aux_dev*link1.z;
     matrix_10.y = f_aux_dev*link1.w;

     matrix_11.x = f_aux_dev*link2.x;
     matrix_11.y = f_aux_dev*link2.y;

     matrix_12.x = f_aux_dev*link2.z;
     matrix_12.y = f_aux_dev*link2.w;

     matrix_20.x = stag_phase*f_aux_dev*C1RE;
     matrix_20.y = stag_phase*f_aux_dev*C1IM;

     matrix_21.x = stag_phase*f_aux_dev*C2RE;
     matrix_21.y = stag_phase*f_aux_dev*C2IM;

     matrix_22.x = stag_phase*f_aux_dev*C3RE;
     matrix_22.y = stag_phase*f_aux_dev*C3IM;

     // matrix=f_aux_dev*U(idx+mu)_nu   

     /////////////////////////////////////////////////////////////////////////////

     site_table[threadIdx.x] = table[idx+(4+nu)*size_dev];                            // idx+nu
     LoadLinkRegs(gauge_texRef, size_dev, site_table[threadIdx.x] + gauge_offset, blockIdx.y);  // U(idx+nu)_mu 
     stag_phase=phases[site_table[threadIdx.x]+blockIdx.y*size_dev];

     mat0.x = matrix_00.x*link0.x+matrix_00.y*link0.y+
              matrix_01.x*link0.z+matrix_01.y*link0.w+
              matrix_02.x*link1.x+matrix_02.y*link1.y;
  
     mat0.y = -matrix_00.x*link0.y+matrix_00.y*link0.x
              -matrix_01.x*link0.w+matrix_01.y*link0.z
              -matrix_02.x*link1.y+matrix_02.y*link1.x;
  
     mat1.x = matrix_00.x*link1.z+matrix_00.y*link1.w+
              matrix_01.x*link2.x+matrix_01.y*link2.y+
              matrix_02.x*link2.z+matrix_02.y*link2.w;

     mat1.y = -matrix_00.x*link1.w+matrix_00.y*link1.z
              -matrix_01.x*link2.y+matrix_01.y*link2.x
              -matrix_02.x*link2.w+matrix_02.y*link2.z;

     mat2.x = stag_phase*(matrix_00.x*C1RE+matrix_00.y*C1IM+
                          matrix_01.x*C2RE+matrix_01.y*C2IM+
                          matrix_02.x*C3RE+matrix_02.y*C3IM);

     mat2.y = stag_phase*(-matrix_00.x*C1IM+matrix_00.y*C1RE
                          -matrix_01.x*C2IM+matrix_01.y*C2RE
                          -matrix_02.x*C3IM+matrix_02.y*C3RE);
  
     matrix_00 = mat0;
     matrix_01 = mat1;
     matrix_02 = mat2;

     mat0.x = matrix_10.x*link0.x+matrix_10.y*link0.y+
              matrix_11.x*link0.z+matrix_11.y*link0.w+
              matrix_12.x*link1.x+matrix_12.y*link1.y;

     mat0.y = -matrix_10.x*link0.y+matrix_10.y*link0.x
              -matrix_11.x*link0.w+matrix_11.y*link0.z
              -matrix_12.x*link1.y+matrix_12.y*link1.x;

     mat1.x = matrix_10.x*link1.z+matrix_10.y*link1.w+
              matrix_11.x*link2.x+matrix_11.y*link2.y+
              matrix_12.x*link2.z+matrix_12.y*link2.w;

     mat1.y = -matrix_10.x*link1.w+matrix_10.y*link1.z
              -matrix_11.x*link2.y+matrix_11.y*link2.x
              -matrix_12.x*link2.w+matrix_12.y*link2.z;

     mat2.x = stag_phase*(matrix_10.x*C1RE+matrix_10.y*C1IM+
                          matrix_11.x*C2RE+matrix_11.y*C2IM+
                          matrix_12.x*C3RE+matrix_12.y*C3IM);

     mat2.y = stag_phase*(-matrix_10.x*C1IM+matrix_10.y*C1RE
                          -matrix_11.x*C2IM+matrix_11.y*C2RE
                          -matrix_12.x*C3IM+matrix_12.y*C3RE);
  
     matrix_10 = mat0;
     matrix_11 = mat1;
     matrix_12 = mat2;
  
     mat0.x = matrix_20.x*link0.x+matrix_20.y*link0.y+
              matrix_21.x*link0.z+matrix_21.y*link0.w+
              matrix_22.x*link1.x+matrix_22.y*link1.y; 
  
     mat0.y = -matrix_20.x*link0.y+matrix_20.y*link0.x
              -matrix_21.x*link0.w+matrix_21.y*link0.z
              -matrix_22.x*link1.y+matrix_22.y*link1.x;

     mat1.x = matrix_20.x*link1.z+matrix_20.y*link1.w+
              matrix_21.x*link2.x+matrix_21.y*link2.y+
              matrix_22.x*link2.z+matrix_22.y*link2.w;
  
     mat1.y = -matrix_20.x*link1.w+matrix_20.y*link1.z
              -matrix_21.x*link2.y+matrix_21.y*link2.x
              -matrix_22.x*link2.w+matrix_22.y*link2.z;
  
     mat2.x = stag_phase*(matrix_20.x*C1RE+matrix_20.y*C1IM+
                          matrix_21.x*C2RE+matrix_21.y*C2IM+
                          matrix_22.x*C3RE+matrix_22.y*C3IM);
  
  
     mat2.y = stag_phase*(-matrix_20.x*C1IM+matrix_20.y*C1RE
                          -matrix_21.x*C2IM+matrix_21.y*C2RE
                          -matrix_22.x*C3IM+matrix_22.y*C3RE);

     matrix_20 = mat0;
     matrix_21 = mat1;
     matrix_22 = mat2;

     // matrix=f_aux_dev*U(idx+mu)_nu * [U(idx+nu)_mu]^{dag}

     __syncthreads();
  
     ////////////////////////////////////////////////////////

     LoadLinkRegs( gauge_texRef, size_dev, idx + gauge_offset, nu);   // U(x)_nu
     stag_phase=phases[idx+nu*size_dev];

     mat0.x = matrix_00.x*link0.x+matrix_00.y*link0.y+
              matrix_01.x*link0.z+matrix_01.y*link0.w+
              matrix_02.x*link1.x+matrix_02.y*link1.y;
  
     mat0.y = -matrix_00.x*link0.y+matrix_00.y*link0.x
              -matrix_01.x*link0.w+matrix_01.y*link0.z
              -matrix_02.x*link1.y+matrix_02.y*link1.x;
  
     mat1.x = matrix_00.x*link1.z+matrix_00.y*link1.w+
              matrix_01.x*link2.x+matrix_01.y*link2.y+
              matrix_02.x*link2.z+matrix_02.y*link2.w;

     mat1.y = -matrix_00.x*link1.w+matrix_00.y*link1.z
              -matrix_01.x*link2.y+matrix_01.y*link2.x
              -matrix_02.x*link2.w+matrix_02.y*link2.z;

     mat2.x = stag_phase*(matrix_00.x*C1RE+matrix_00.y*C1IM+
                          matrix_01.x*C2RE+matrix_01.y*C2IM+
                          matrix_02.x*C3RE+matrix_02.y*C3IM);

     mat2.y = stag_phase*(-matrix_00.x*C1IM+matrix_00.y*C1RE
                          -matrix_01.x*C2IM+matrix_01.y*C2RE
                          -matrix_02.x*C3IM+matrix_02.y*C3RE);

     matrix_00 = mat0;
     matrix_01 = mat1;
     matrix_02 = mat2;

     mat0.x = matrix_10.x*link0.x+matrix_10.y*link0.y+
              matrix_11.x*link0.z+matrix_11.y*link0.w+
              matrix_12.x*link1.x+matrix_12.y*link1.y;

     mat0.y = -matrix_10.x*link0.y+matrix_10.y*link0.x
              -matrix_11.x*link0.w+matrix_11.y*link0.z
              -matrix_12.x*link1.y+matrix_12.y*link1.x;

     mat1.x = matrix_10.x*link1.z+matrix_10.y*link1.w+
              matrix_11.x*link2.x+matrix_11.y*link2.y+
              matrix_12.x*link2.z+matrix_12.y*link2.w;

     mat1.y = -matrix_10.x*link1.w+matrix_10.y*link1.z
              -matrix_11.x*link2.y+matrix_11.y*link2.x
             -matrix_12.x*link2.w+matrix_12.y*link2.z;

     mat2.x = stag_phase*(matrix_10.x*C1RE+matrix_10.y*C1IM+
                          matrix_11.x*C2RE+matrix_11.y*C2IM+
                          matrix_12.x*C3RE+matrix_12.y*C3IM);

     mat2.y = stag_phase*(-matrix_10.x*C1IM+matrix_10.y*C1RE
                          -matrix_11.x*C2IM+matrix_11.y*C2RE
                          -matrix_12.x*C3IM+matrix_12.y*C3RE);
 
     matrix_10 = mat0;
     matrix_11 = mat1;
     matrix_12 = mat2;

     mat0.x = matrix_20.x*link0.x+matrix_20.y*link0.y+
              matrix_21.x*link0.z+matrix_21.y*link0.w+
              matrix_22.x*link1.x+matrix_22.y*link1.y; 
  
     mat0.y = -matrix_20.x*link0.y+matrix_20.y*link0.x
              -matrix_21.x*link0.w+matrix_21.y*link0.z
              -matrix_22.x*link1.y+matrix_22.y*link1.x;

     mat1.x = matrix_20.x*link1.z+matrix_20.y*link1.w+
              matrix_21.x*link2.x+matrix_21.y*link2.y+
              matrix_22.x*link2.z+matrix_22.y*link2.w;
  
     mat1.y = -matrix_20.x*link1.w+matrix_20.y*link1.z
              -matrix_21.x*link2.y+matrix_21.y*link2.x
              -matrix_22.x*link2.w+matrix_22.y*link2.z;
  
     mat2.x = stag_phase*(matrix_20.x*C1RE+matrix_20.y*C1IM+
                          matrix_21.x*C2RE+matrix_21.y*C2IM+
                          matrix_22.x*C3RE+matrix_22.y*C3IM);
  
     mat2.y = stag_phase*(-matrix_20.x*C1IM+matrix_20.y*C1RE
                          -matrix_21.x*C2IM+matrix_21.y*C2RE
                          -matrix_22.x*C3IM+matrix_22.y*C3RE);

     matrix_20 = mat0;
     matrix_21 = mat1;
     matrix_22 = mat2;

     // matrix=f_aux_dev*U(idx+mu)_nu * [U(idx+nu)_mu]^{dag} * [U(x)_nu]^{dag}

     __syncthreads();

     /////////////////////////// End of forward staples

     /// Write to global memory
     staples[0][threadIdx.x].x += matrix_00.x;
     staples[0][threadIdx.x].y += matrix_00.y;
     staples[1][threadIdx.x].x += matrix_01.x;
     staples[1][threadIdx.x].y += matrix_01.y;
     staples[2][threadIdx.x].x += matrix_02.x;
     staples[2][threadIdx.x].y += matrix_02.y;

     staples[3][threadIdx.x].x += matrix_10.x;
     staples[3][threadIdx.x].y += matrix_10.y;
     staples[4][threadIdx.x].x += matrix_11.x;
     staples[4][threadIdx.x].y += matrix_11.y;
     staples[5][threadIdx.x].x += matrix_12.x;
     staples[5][threadIdx.x].y += matrix_12.y;

     staples[6][threadIdx.x].x += matrix_20.x;
     staples[6][threadIdx.x].y += matrix_20.y;
     staples[7][threadIdx.x].x += matrix_21.x;
     staples[7][threadIdx.x].y += matrix_21.y;
     staples[8][threadIdx.x].x += matrix_22.x;
     staples[8][threadIdx.x].y += matrix_22.y;

     ///////////////////////////////////////////////

//             (1)
//           +--<--+
//           |     |
// ^mu   (2) V     |   to calculate (1)*(2)*(3)
// |         |     |
//           +-->--+
// ->nu  temp  (3)  idx

     temp_site = table[idx+nu*size_dev];
     site_table[threadIdx.x]     = table[temp_site+(4+blockIdx.y)*size_dev]; 
     LoadLinkRegs(gauge_texRef, size_dev, site_table[threadIdx.x] + gauge_offset, nu);  // U(idx-nu+mu)_{nu}
     stag_phase=phases[site_table[threadIdx.x]+nu*size_dev];
 
     matrix_00.x =  f_aux_dev*link0.x;
     matrix_00.y = -f_aux_dev*link0.y;

     matrix_01.x =  f_aux_dev*link1.z;
     matrix_01.y = -f_aux_dev*link1.w;

     matrix_02.x =  stag_phase*f_aux_dev*C1RE;
     matrix_02.y = -stag_phase*f_aux_dev*C1IM;

     matrix_10.x =  f_aux_dev*link0.z;
     matrix_10.y = -f_aux_dev*link0.w;

     matrix_11.x =  f_aux_dev*link2.x;
     matrix_11.y = -f_aux_dev*link2.y;

     matrix_12.x =  stag_phase*f_aux_dev*C2RE;
     matrix_12.y = -stag_phase*f_aux_dev*C2IM;

     matrix_20.x =  f_aux_dev*link1.x;
     matrix_20.y = -f_aux_dev*link1.y;

     matrix_21.x =  f_aux_dev*link2.z;
     matrix_21.y = -f_aux_dev*link2.w;

     matrix_22.x =  stag_phase*f_aux_dev*C3RE;
     matrix_22.y = -stag_phase*f_aux_dev*C3IM;

     // matrix=f_aux_dev [U(idx-nu+mu)_{nu}]^{dag}

     ///////////////////////////////////////////////

     LoadLinkRegs(gauge_texRef, size_dev, temp_site + gauge_offset, blockIdx.y); // U(idx-nu)_mu
     stag_phase=phases[temp_site+blockIdx.y*size_dev];

     mat0.x = matrix_00.x*link0.x+matrix_00.y*link0.y+
              matrix_01.x*link0.z+matrix_01.y*link0.w+
              matrix_02.x*link1.x+matrix_02.y*link1.y;
  
     mat0.y = -matrix_00.x*link0.y+matrix_00.y*link0.x
              -matrix_01.x*link0.w+matrix_01.y*link0.z
              -matrix_02.x*link1.y+matrix_02.y*link1.x;
  
     mat1.x = matrix_00.x*link1.z+matrix_00.y*link1.w+
              matrix_01.x*link2.x+matrix_01.y*link2.y+
              matrix_02.x*link2.z+matrix_02.y*link2.w;

     mat1.y = -matrix_00.x*link1.w+matrix_00.y*link1.z
              -matrix_01.x*link2.y+matrix_01.y*link2.x
              -matrix_02.x*link2.w+matrix_02.y*link2.z;

     mat2.x = stag_phase*(matrix_00.x*C1RE+matrix_00.y*C1IM+
                          matrix_01.x*C2RE+matrix_01.y*C2IM+
                          matrix_02.x*C3RE+matrix_02.y*C3IM);

     mat2.y = stag_phase*(-matrix_00.x*C1IM+matrix_00.y*C1RE
                          -matrix_01.x*C2IM+matrix_01.y*C2RE
                          -matrix_02.x*C3IM+matrix_02.y*C3RE);
  
     matrix_00 = mat0;
     matrix_01 = mat1;
     matrix_02 = mat2;

     mat0.x = matrix_10.x*link0.x+matrix_10.y*link0.y+
              matrix_11.x*link0.z+matrix_11.y*link0.w+
              matrix_12.x*link1.x+matrix_12.y*link1.y;

     mat0.y = -matrix_10.x*link0.y+matrix_10.y*link0.x
              -matrix_11.x*link0.w+matrix_11.y*link0.z
              -matrix_12.x*link1.y+matrix_12.y*link1.x;

     mat1.x = matrix_10.x*link1.z+matrix_10.y*link1.w+
              matrix_11.x*link2.x+matrix_11.y*link2.y+
              matrix_12.x*link2.z+matrix_12.y*link2.w;

     mat1.y = -matrix_10.x*link1.w+matrix_10.y*link1.z
              -matrix_11.x*link2.y+matrix_11.y*link2.x
              -matrix_12.x*link2.w+matrix_12.y*link2.z;

     mat2.x = stag_phase*(matrix_10.x*C1RE+matrix_10.y*C1IM+
                          matrix_11.x*C2RE+matrix_11.y*C2IM+
                          matrix_12.x*C3RE+matrix_12.y*C3IM);

     mat2.y = stag_phase*(-matrix_10.x*C1IM+matrix_10.y*C1RE
                          -matrix_11.x*C2IM+matrix_11.y*C2RE
                          -matrix_12.x*C3IM+matrix_12.y*C3RE);
  
     matrix_10 = mat0;
     matrix_11 = mat1;
     matrix_12 = mat2;
  
     mat0.x = matrix_20.x*link0.x+matrix_20.y*link0.y+
              matrix_21.x*link0.z+matrix_21.y*link0.w+
              matrix_22.x*link1.x+matrix_22.y*link1.y; 
  
     mat0.y = -matrix_20.x*link0.y+matrix_20.y*link0.x
              -matrix_21.x*link0.w+matrix_21.y*link0.z
              -matrix_22.x*link1.y+matrix_22.y*link1.x;

     mat1.x = matrix_20.x*link1.z+matrix_20.y*link1.w+
              matrix_21.x*link2.x+matrix_21.y*link2.y+
              matrix_22.x*link2.z+matrix_22.y*link2.w;
  
     mat1.y = -matrix_20.x*link1.w+matrix_20.y*link1.z
              -matrix_21.x*link2.y+matrix_21.y*link2.x
              -matrix_22.x*link2.w+matrix_22.y*link2.z;
  
     mat2.x =  stag_phase*(matrix_20.x*C1RE+matrix_20.y*C1IM+
                           matrix_21.x*C2RE+matrix_21.y*C2IM+
                           matrix_22.x*C3RE+matrix_22.y*C3IM);
  
     mat2.y = stag_phase*(-matrix_20.x*C1IM+matrix_20.y*C1RE
                          -matrix_21.x*C2IM+matrix_21.y*C2RE
                          -matrix_22.x*C3IM+matrix_22.y*C3RE);

     matrix_20 = mat0;
     matrix_21 = mat1;
     matrix_22 = mat2;

     // matrix=f_aux_dev [U(idx-nu+mu)_{nu}]^{dag} * [U(idx-nu)_mu]^{dag}

     __syncthreads(); 

     ///////////////////////////////////////////////

     LoadLinkRegs(gauge_texRef, size_dev, temp_site + gauge_offset, nu); // U(x-nu)_nu
     stag_phase=phases[temp_site+nu*size_dev];

     mat0.x =matrix_00.x*link0.x-matrix_00.y*link0.y+
             matrix_01.x*link1.z-matrix_01.y*link1.w+
             stag_phase*(matrix_02.x*C1RE   -matrix_02.y*C1IM);
  
     mat0.y = matrix_00.x*link0.y+matrix_00.y*link0.x+
              matrix_01.x*link1.w+matrix_01.y*link1.z+
              stag_phase*(matrix_02.x*C1IM   +matrix_02.y*C1RE);

     mat1.x = matrix_00.x*link0.z-matrix_00.y*link0.w+
              matrix_01.x*link2.x-matrix_01.y*link2.y+
              stag_phase*(matrix_02.x*C2RE   -matrix_02.y*C2IM);
  
     mat1.y = matrix_00.x*link0.w+matrix_00.y*link0.z+
              matrix_01.x*link2.y+matrix_01.y*link2.x+
              stag_phase*(matrix_02.x*C2IM   +matrix_02.y*C2RE);

     mat2.x = matrix_00.x*link1.x-matrix_00.y*link1.y+
              matrix_01.x*link2.z-matrix_01.y*link2.w+
              stag_phase*(matrix_02.x*C3RE   -matrix_02.y*C3IM);
  
     mat2.y = matrix_00.x*link1.y+matrix_00.y*link1.x+
              matrix_01.x*link2.w+matrix_01.y*link2.z+
              stag_phase*(matrix_02.x*C3IM   +matrix_02.y*C3RE);

     matrix_00 = mat0;
     matrix_01 = mat1;
     matrix_02 = mat2;

     mat0.x = matrix_10.x*link0.x-matrix_10.y*link0.y+
              matrix_11.x*link1.z-matrix_11.y*link1.w+
              stag_phase*(matrix_12.x*C1RE   -matrix_12.y*C1IM);
  
     mat0.y = matrix_10.x*link0.y+matrix_10.y*link0.x+
              matrix_11.x*link1.w+matrix_11.y*link1.z+
              stag_phase*(matrix_12.x*C1IM   +matrix_12.y*C1RE);

     mat1.x = matrix_10.x*link0.z-matrix_10.y*link0.w+
              matrix_11.x*link2.x-matrix_11.y*link2.y+
              stag_phase*(matrix_12.x*C2RE   -matrix_12.y*C2IM);
  
     mat1.y = matrix_10.x*link0.w+matrix_10.y*link0.z+
              matrix_11.x*link2.y+matrix_11.y*link2.x+
              stag_phase*(matrix_12.x*C2IM   +matrix_12.y*C2RE);

     mat2.x = matrix_10.x*link1.x-matrix_10.y*link1.y+
              matrix_11.x*link2.z-matrix_11.y*link2.w+
              stag_phase*(matrix_12.x*C3RE   -matrix_12.y*C3IM);
  
     mat2.y = matrix_10.x*link1.y+matrix_10.y*link1.x+
              matrix_11.x*link2.w+matrix_11.y*link2.z+
              stag_phase*(matrix_12.x*C3IM   +matrix_12.y*C3RE);

     matrix_10 = mat0;
     matrix_11 = mat1;
     matrix_12 = mat2;

     mat0.x = matrix_20.x*link0.x-matrix_20.y*link0.y+
              matrix_21.x*link1.z-matrix_21.y*link1.w+
              stag_phase*(matrix_22.x*C1RE   -matrix_22.y*C1IM);
  
     mat0.y = matrix_20.x*link0.y+matrix_20.y*link0.x+
              matrix_21.x*link1.w+matrix_21.y*link1.z+
              stag_phase*(matrix_22.x*C1IM   +matrix_22.y*C1RE);

     mat1.x = matrix_20.x*link0.z-matrix_20.y*link0.w+
              matrix_21.x*link2.x-matrix_21.y*link2.y+
              stag_phase*(matrix_22.x*C2RE   -matrix_22.y*C2IM);
  
     mat1.y = matrix_20.x*link0.w+matrix_20.y*link0.z+
              matrix_21.x*link2.y+matrix_21.y*link2.x+
              stag_phase*(matrix_22.x*C2IM   +matrix_22.y*C2RE);

     mat2.x = matrix_20.x*link1.x-matrix_20.y*link1.y+
              matrix_21.x*link2.z-matrix_21.y*link2.w+
              stag_phase*(matrix_22.x*C3RE   -matrix_22.y*C3IM);
  
     mat2.y = matrix_20.x*link1.y+matrix_20.y*link1.x+
              matrix_21.x*link2.w+matrix_21.y*link2.z+
              stag_phase*(matrix_22.x*C3IM   +matrix_22.y*C3RE);

     matrix_20 = mat0;
     matrix_21 = mat1;
     matrix_22 = mat2;

     // matrix=f_aux_dev [U(idx-nu+mu)_{nu}]^{dag} * [U(idx-nu)_mu]^{dag} * U(x-nu)_nu

     __syncthreads();   


     /////////////////////////////

     /// Write to global memory
     staples[0][threadIdx.x].x += matrix_00.x;
     staples[0][threadIdx.x].y += matrix_00.y;
     staples[1][threadIdx.x].x += matrix_01.x;
     staples[1][threadIdx.x].y += matrix_01.y;
     staples[2][threadIdx.x].x += matrix_02.x;
     staples[2][threadIdx.x].y += matrix_02.y;

     staples[3][threadIdx.x].x += matrix_10.x;
     staples[3][threadIdx.x].y += matrix_10.y;
     staples[4][threadIdx.x].x += matrix_11.x;
     staples[4][threadIdx.x].y += matrix_11.y;
     staples[5][threadIdx.x].x += matrix_12.x;
     staples[5][threadIdx.x].y += matrix_12.y;

     staples[6][threadIdx.x].x += matrix_20.x;
     staples[6][threadIdx.x].y += matrix_20.y;
     staples[7][threadIdx.x].x += matrix_21.x;
     staples[7][threadIdx.x].y += matrix_21.y;
     staples[8][threadIdx.x].x += matrix_22.x;
     staples[8][threadIdx.x].y += matrix_22.y;
     }

  ///////////////////////////////////////////

  // Load out_matrix
  stag_phase=phases[idx+blockIdx.y*size_dev];

  matrix_00.x = staples[0][threadIdx.x].x;
  matrix_00.y = staples[0][threadIdx.x].y;
  matrix_01.x = staples[1][threadIdx.x].x;
  matrix_01.y = staples[1][threadIdx.x].y;
  matrix_02.x = staples[2][threadIdx.x].x;
  matrix_02.y = staples[2][threadIdx.x].y;

  matrix_10.x = staples[3][threadIdx.x].x;
  matrix_10.y = staples[3][threadIdx.x].y;
  matrix_11.x = staples[4][threadIdx.x].x;
  matrix_11.y = staples[4][threadIdx.x].y;
  matrix_12.x = staples[5][threadIdx.x].x;
  matrix_12.y = staples[5][threadIdx.x].y;

  matrix_20.x = staples[6][threadIdx.x].x;
  matrix_20.y = staples[6][threadIdx.x].y;
  matrix_21.x = staples[7][threadIdx.x].x;
  matrix_21.y = staples[7][threadIdx.x].y;
  matrix_22.x = staples[8][threadIdx.x].x;
  matrix_22.y = staples[8][threadIdx.x].y;

  //////////////////

  // Multiply u_mu * staple
  LoadLinkRegs(gauge_texRef, size_dev, idx + gauge_offset, blockIdx.y);   //Loads U_mu

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

  /////////////////////////////

  /// Write to global memory the traceless antihermitian part
  out_matrix[idx +            2*blockIdx.y*size_dev].x+=0.5f*(matrix_01.x-matrix_10.x);
  out_matrix[idx +            2*blockIdx.y*size_dev].y+=0.5f*(matrix_01.y+matrix_10.y);
  out_matrix[idx +            2*blockIdx.y*size_dev].z+=0.5f*(matrix_02.x-matrix_20.x);
  out_matrix[idx +            2*blockIdx.y*size_dev].w+=0.5f*(matrix_02.y+matrix_20.y);

  out_matrix[idx + size_dev + 2*blockIdx.y*size_dev].x+=0.5f*(matrix_12.x-matrix_21.x);
  out_matrix[idx + size_dev + 2*blockIdx.y*size_dev].y+=0.5f*(matrix_12.y+matrix_21.y);
  out_matrix[idx + size_dev + 2*blockIdx.y*size_dev].z+=matrix_00.y-0.3333333f*(matrix_00.y+matrix_11.y+matrix_22.y);
  out_matrix[idx + size_dev + 2*blockIdx.y*size_dev].w+=matrix_11.y-0.3333333f*(matrix_00.y+matrix_11.y+matrix_22.y);
  }



/////////////////////////////////////////////////////////////////////////////////////////


extern "C" void  cuda_gauge_deriv(int ipdot_init)
  {
  #ifdef DEBUG_MODE
  printf("DEBUG: inside cuda_gauge_deriv ...\n");
  #endif

  size_t gauge_field_size = sizeof(float)*no_links*12;

  //Copy to device constant memory
  float factor = (float) 1.0*beta/3.0;
  cudaSafe(AT,cudaMemcpyToSymbol(f_aux_dev, &factor, sizeof(float), 0, cudaMemcpyHostToDevice), "cudaMemcpyToSymbol");

  // initialize ipdot_device to zero if ipdot_init==0
  if(ipdot_init==0) cudaSafe(AT,cudaMemset(ipdot_device, 0, 8*no_links*sizeof(float)), "cudaMemset");

  //Setting block and grid sizes for the kernel
  dim3 DerivBlockDimension(NUM_THREADS); 
  dim3 GridDimension(size/DerivBlockDimension.x,4); //Run separately the four dimensions   

  size_t offset_g;
  cudaSafe(AT,cudaBindTexture(&offset_g, gauge_texRef, gauge_field_device, gauge_field_size), "cudaBindTexture");
  offset_g/=sizeof(float4);

  //Kernel execution
  DeriveFieldKernel<<<GridDimension,DerivBlockDimension>>>(ipdot_device,
						           device_table,
                                                           device_phases,
                                                           offset_g);
  cudaCheckError(AT,"DeriveFieldKernel");  

  cudaSafe(AT,cudaUnbindTexture(gauge_texRef), "cudaUnbindTexture");

  #ifdef PARAMETER_TEST
    double *norm2_dev, norm2_host;
    cudaSafe(AT,cudaMalloc((void**)&norm2_dev, sizeof(double)), "cudaMalloc"); 

    IpdotNorm2(norm2_dev, ipdot_device);
 
    cudaSafe(AT,cudaMemcpy(&norm2_host, norm2_dev, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy"); 
    printf("L2 norm of GAUGE force = %g\n", sqrt(norm2_host));

    cudaSafe(AT,cudaFree(norm2_dev), "cudaFree");
  #endif

  #ifdef DEBUG_MODE
  printf("\tterminated cuda_gauge_deriv\n");
  #endif
  }
