#include "include/global_macro.h"


__global__ void DoeKernel(float2  *in,
                          float2  *out,
                          int *tables, 
                          int *phases,
                          size_t gauge_offset,
                          size_t ferm_offset) 
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stag_phase = 1;

  //Store result in sharedMem
  __shared__ float ferm_out[3][2][NUM_THREADS];

  //New tables indexing (index fastest)
  __shared__ int site_table[NUM_THREADS];

  //Load link matrix U_mu(ix) in registers
  DeclareMatrixRegs;      //12 registers

  float2 ferm_in_0, ferm_in_1, ferm_in_2;

  #ifdef IM_CHEM_POT
   float2 ferm_aux_0, ferm_aux_1, ferm_aux_2;
  #endif
 
  // Direction 0
  site_table[threadIdx.x]  = tables[idx+4*size_dev];

  ferm_in_0 = tex1Dfetch(fermion_texRef,              site_table[threadIdx.x] + ferm_offset);
  ferm_in_1 = tex1Dfetch(fermion_texRef,   size_dev + site_table[threadIdx.x] + ferm_offset);
  ferm_in_2 = tex1Dfetch(fermion_texRef, 2*size_dev + site_table[threadIdx.x] + ferm_offset);
 
  LoadLinkRegs(gauge_texRef, size_dev, idx+gauge_offset, 0);

  ferm_out[0][0][threadIdx.x] = link0.x*ferm_in_0.x-link0.y*ferm_in_0.y+  
                                link0.z*ferm_in_1.x-link0.w*ferm_in_1.y+ 
                                link1.x*ferm_in_2.x-link1.y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] = link0.x*ferm_in_0.y+link0.y*ferm_in_0.x+ 
                                link0.z*ferm_in_1.y+link0.w*ferm_in_1.x+ 
                                link1.x*ferm_in_2.y+link1.y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] = link1.z*ferm_in_0.x-link1.w*ferm_in_0.y+  
                                link2.x*ferm_in_1.x-link2.y*ferm_in_1.y+ 
                                link2.z*ferm_in_2.x-link2.w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] = link1.z*ferm_in_0.y+link1.w*ferm_in_0.x+ 
                                link2.x*ferm_in_1.y+link2.y*ferm_in_1.x+ 
                                link2.z*ferm_in_2.y+link2.w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] = C1RE*ferm_in_0.x-C1IM*ferm_in_0.y+  
                                C2RE*ferm_in_1.x-C2IM*ferm_in_1.y+ 
                                C3RE*ferm_in_2.x-C3IM*ferm_in_2.y; 
  ferm_out[2][1][threadIdx.x] = C1RE*ferm_in_0.y+C1IM*ferm_in_0.x+ 
                                C2RE*ferm_in_1.y+C2IM*ferm_in_1.x+ 
                                C3RE*ferm_in_2.y+C3IM*ferm_in_2.x; 

  //Direction 1
  site_table[threadIdx.x] = tables[idx+5*size_dev];
  stag_phase              = phases[idx+size_dev];

  ferm_in_0 = tex1Dfetch(fermion_texRef,              site_table[threadIdx.x] + ferm_offset);
  ferm_in_1 = tex1Dfetch(fermion_texRef,   size_dev + site_table[threadIdx.x] + ferm_offset);
  ferm_in_2 = tex1Dfetch(fermion_texRef, 2*size_dev + site_table[threadIdx.x] + ferm_offset);

  LoadLinkRegs(gauge_texRef, size_dev, idx+gauge_offset, 1);

  ferm_out[0][0][threadIdx.x] += link0.x*ferm_in_0.x-link0.y*ferm_in_0.y+  
                                 link0.z*ferm_in_1.x-link0.w*ferm_in_1.y+ 
                                 link1.x*ferm_in_2.x-link1.y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0.x*ferm_in_0.y+link0.y*ferm_in_0.x+ 
                                 link0.z*ferm_in_1.y+link0.w*ferm_in_1.x+ 
                                 link1.x*ferm_in_2.y+link1.y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1.z*ferm_in_0.x-link1.w*ferm_in_0.y+  
                                 link2.x*ferm_in_1.x-link2.y*ferm_in_1.y+ 
                                 link2.z*ferm_in_2.x-link2.w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1.z*ferm_in_0.y+link1.w*ferm_in_0.x+ 
                                 link2.x*ferm_in_1.y+link2.y*ferm_in_1.x+ 
                                 link2.z*ferm_in_2.y+link2.w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RE*ferm_in_0.x-C1IM*ferm_in_0.y+  
					     C2RE*ferm_in_1.x-C2IM*ferm_in_1.y+ 
					     C3RE*ferm_in_2.x-C3IM*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RE*ferm_in_0.y+C1IM*ferm_in_0.x+ 
					     C2RE*ferm_in_1.y+C2IM*ferm_in_1.x+ 
					     C3RE*ferm_in_2.y+C3IM*ferm_in_2.x); 

  //Direction 2
  site_table[threadIdx.x] = tables[idx+6*size_dev];
  stag_phase              = phases[idx+2*size_dev];

  ferm_in_0 = tex1Dfetch(fermion_texRef,              site_table[threadIdx.x] + ferm_offset);
  ferm_in_1 = tex1Dfetch(fermion_texRef,   size_dev + site_table[threadIdx.x] + ferm_offset);
  ferm_in_2 = tex1Dfetch(fermion_texRef, 2*size_dev + site_table[threadIdx.x] + ferm_offset);

  LoadLinkRegs(gauge_texRef, size_dev, idx+gauge_offset, 2);

  ferm_out[0][0][threadIdx.x] += link0.x*ferm_in_0.x-link0.y*ferm_in_0.y+  
                                 link0.z*ferm_in_1.x-link0.w*ferm_in_1.y+ 
                                 link1.x*ferm_in_2.x-link1.y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0.x*ferm_in_0.y+link0.y*ferm_in_0.x+ 
                                 link0.z*ferm_in_1.y+link0.w*ferm_in_1.x+ 
                                 link1.x*ferm_in_2.y+link1.y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1.z*ferm_in_0.x-link1.w*ferm_in_0.y+  
                                 link2.x*ferm_in_1.x-link2.y*ferm_in_1.y+ 
                                 link2.z*ferm_in_2.x-link2.w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1.z*ferm_in_0.y+link1.w*ferm_in_0.x+ 
                                 link2.x*ferm_in_1.y+link2.y*ferm_in_1.x+ 
                                 link2.z*ferm_in_2.y+link2.w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RE*ferm_in_0.x-C1IM*ferm_in_0.y+  
					     C2RE*ferm_in_1.x-C2IM*ferm_in_1.y+ 
					     C3RE*ferm_in_2.x-C3IM*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RE*ferm_in_0.y+C1IM*ferm_in_0.x+ 
					     C2RE*ferm_in_1.y+C2IM*ferm_in_1.x+ 
					     C3RE*ferm_in_2.y+C3IM*ferm_in_2.x); 
  
  
  //Direction 3
  site_table[threadIdx.x]  = tables[idx+7*size_dev];
  stag_phase               = phases[idx+3*size_dev];

  ferm_in_0 = tex1Dfetch(fermion_texRef,              site_table[threadIdx.x] + ferm_offset);
  ferm_in_1 = tex1Dfetch(fermion_texRef,   size_dev + site_table[threadIdx.x] + ferm_offset);
  ferm_in_2 = tex1Dfetch(fermion_texRef, 2*size_dev + site_table[threadIdx.x] + ferm_offset);

  LoadLinkRegs(gauge_texRef, size_dev, idx+gauge_offset, 3);

  #ifndef IM_CHEM_POT
  ferm_out[0][0][threadIdx.x] += link0.x*ferm_in_0.x-link0.y*ferm_in_0.y+  
                                 link0.z*ferm_in_1.x-link0.w*ferm_in_1.y+ 
                                 link1.x*ferm_in_2.x-link1.y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0.x*ferm_in_0.y+link0.y*ferm_in_0.x+ 
                                 link0.z*ferm_in_1.y+link0.w*ferm_in_1.x+ 
                                 link1.x*ferm_in_2.y+link1.y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1.z*ferm_in_0.x-link1.w*ferm_in_0.y+  
                                 link2.x*ferm_in_1.x-link2.y*ferm_in_1.y+ 
                                 link2.z*ferm_in_2.x-link2.w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1.z*ferm_in_0.y+link1.w*ferm_in_0.x+ 
                                 link2.x*ferm_in_1.y+link2.y*ferm_in_1.x+ 
                                 link2.z*ferm_in_2.y+link2.w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RE*ferm_in_0.x-C1IM*ferm_in_0.y+  
					     C2RE*ferm_in_1.x-C2IM*ferm_in_1.y+ 
					     C3RE*ferm_in_2.x-C3IM*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RE*ferm_in_0.y+C1IM*ferm_in_0.x+ 
					     C2RE*ferm_in_1.y+C2IM*ferm_in_1.x+ 
					     C3RE*ferm_in_2.y+C3IM*ferm_in_2.x); 
  #else
  ferm_aux_0.x = link0.x*ferm_in_0.x-link0.y*ferm_in_0.y+  
                 link0.z*ferm_in_1.x-link0.w*ferm_in_1.y+ 
                 link1.x*ferm_in_2.x-link1.y*ferm_in_2.y; 
  ferm_aux_0.y = link0.x*ferm_in_0.y+link0.y*ferm_in_0.x+ 
                 link0.z*ferm_in_1.y+link0.w*ferm_in_1.x+ 
                 link1.x*ferm_in_2.y+link1.y*ferm_in_2.x; 

  ferm_aux_1.x = link1.z*ferm_in_0.x-link1.w*ferm_in_0.y+  
                 link2.x*ferm_in_1.x-link2.y*ferm_in_1.y+ 
                 link2.z*ferm_in_2.x-link2.w*ferm_in_2.y; 
  ferm_aux_1.y = link1.z*ferm_in_0.y+link1.w*ferm_in_0.x+ 
                 link2.x*ferm_in_1.y+link2.y*ferm_in_1.x+ 
                 link2.z*ferm_in_2.y+link2.w*ferm_in_2.x; 

  ferm_aux_2.x = stag_phase*(C1RE*ferm_in_0.x-C1IM*ferm_in_0.y+  
	         C2RE*ferm_in_1.x-C2IM*ferm_in_1.y+ 
	         C3RE*ferm_in_2.x-C3IM*ferm_in_2.y); 
  ferm_aux_2.y = stag_phase*(C1RE*ferm_in_0.y+C1IM*ferm_in_0.x+ 
	         C2RE*ferm_in_1.y+C2IM*ferm_in_1.x+ 
	         C3RE*ferm_in_2.y+C3IM*ferm_in_2.x); 

  ferm_out[0][0][threadIdx.x] += ferm_aux_0.x*dev_eim_cos_f - ferm_aux_0.y*dev_eim_sin_f;  // Re[e^{imu}*ferm_aux_0]
  ferm_out[0][1][threadIdx.x] += ferm_aux_0.x*dev_eim_sin_f + ferm_aux_0.y*dev_eim_cos_f;  // Im[e^{imu}*ferm_aux_0]

  ferm_out[1][0][threadIdx.x] += ferm_aux_1.x*dev_eim_cos_f - ferm_aux_1.y*dev_eim_sin_f;  // Re[e^{imu}*ferm_aux_1]
  ferm_out[1][1][threadIdx.x] += ferm_aux_1.x*dev_eim_sin_f + ferm_aux_1.y*dev_eim_cos_f;  // Im[e^{imu}*ferm_aux_1]

  ferm_out[2][0][threadIdx.x] += ferm_aux_2.x*dev_eim_cos_f - ferm_aux_2.y*dev_eim_sin_f;  // Re[e^{imu}*ferm_aux_2]
  ferm_out[2][1][threadIdx.x] += ferm_aux_2.x*dev_eim_sin_f + ferm_aux_2.y*dev_eim_cos_f;  // Im[e^{imu}*ferm_aux_2]
  #endif

  
  //---------------------------------------------------end of first block
 
  //Direction 0
  site_table[threadIdx.x] = tables[idx];
 
  ferm_in_0 = tex1Dfetch(fermion_texRef,              site_table[threadIdx.x] + ferm_offset);
  ferm_in_1 = tex1Dfetch(fermion_texRef,   size_dev + site_table[threadIdx.x] + ferm_offset);
  ferm_in_2 = tex1Dfetch(fermion_texRef, 2*size_dev + site_table[threadIdx.x] + ferm_offset);

  LoadLinkRegs(gauge_texRef, size_dev, site_table[threadIdx.x]+gauge_offset, 0);

  ferm_out[0][0][threadIdx.x] -= link0.x*ferm_in_0.x+link0.y*ferm_in_0.y +
              			 link1.z*ferm_in_1.x+link1.w*ferm_in_1.y +
				 C1RE*ferm_in_2.x   +C1IM*ferm_in_2.y; 
  
  ferm_out[0][1][threadIdx.x] -= link0.x*ferm_in_0.y-link0.y*ferm_in_0.x +
                                 link1.z*ferm_in_1.y-link1.w*ferm_in_1.x +
                                 C1RE*ferm_in_2.y   -C1IM*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] -= link0.z*ferm_in_0.x+link0.w*ferm_in_0.y +
                                 link2.x*ferm_in_1.x+link2.y*ferm_in_1.y +
                                 C2RE*ferm_in_2.x   +C2IM*ferm_in_2.y; 

  ferm_out[1][1][threadIdx.x] -= link0.z*ferm_in_0.y-link0.w*ferm_in_0.x +
                                 link2.x*ferm_in_1.y-link2.y*ferm_in_1.x +
                                 C2RE*ferm_in_2.y   -C2IM*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] -= link1.x*ferm_in_0.x+link1.y*ferm_in_0.y +
                                 link2.z*ferm_in_1.x+link2.w*ferm_in_1.y +
                                 C3RE*ferm_in_2.x   +C3IM*ferm_in_2.y; 

  ferm_out[2][1][threadIdx.x] -= link1.x*ferm_in_0.y-link1.y*ferm_in_0.x +
                                 link2.z*ferm_in_1.y-link2.w*ferm_in_1.x +
                                 C3RE*ferm_in_2.y   -C3IM*ferm_in_2.x; 

  
  //Direction 1
  site_table[threadIdx.x] = tables[idx+size_dev];
  stag_phase              = phases[site_table[threadIdx.x]+size_dev];

  ferm_in_0 = tex1Dfetch(fermion_texRef,              site_table[threadIdx.x] + ferm_offset);
  ferm_in_1 = tex1Dfetch(fermion_texRef,   size_dev + site_table[threadIdx.x] + ferm_offset);
  ferm_in_2 = tex1Dfetch(fermion_texRef, 2*size_dev + site_table[threadIdx.x] + ferm_offset);

  LoadLinkRegs(gauge_texRef, size_dev, site_table[threadIdx.x]+gauge_offset, 1);

  ferm_out[0][0][threadIdx.x] -= link0.x*ferm_in_0.x+link0.y*ferm_in_0.y +
                                 link1.z*ferm_in_1.x+link1.w*ferm_in_1.y +
                                 stag_phase*(C1RE*ferm_in_2.x+C1IM*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0.x*ferm_in_0.y-link0.y*ferm_in_0.x +
                                 link1.z*ferm_in_1.y-link1.w*ferm_in_1.x +
                                 stag_phase*(C1RE*ferm_in_2.y-C1IM*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0.z*ferm_in_0.x+link0.w*ferm_in_0.y +
                                 link2.x*ferm_in_1.x+link2.y*ferm_in_1.y +
                                 stag_phase*(C2RE*ferm_in_2.x+C2IM*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0.z*ferm_in_0.y-link0.w*ferm_in_0.x +
                                 link2.x*ferm_in_1.y-link2.y*ferm_in_1.x +
                                 stag_phase*(C2RE*ferm_in_2.y-C2IM*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1.x*ferm_in_0.x+link1.y*ferm_in_0.y +
                                 link2.z*ferm_in_1.x+link2.w*ferm_in_1.y +
                                 stag_phase*(C3RE*ferm_in_2.x+C3IM*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1.x*ferm_in_0.y-link1.y*ferm_in_0.x +
                                 link2.z*ferm_in_1.y-link2.w*ferm_in_1.x +
                                 stag_phase*(C3RE*ferm_in_2.y- C3IM*ferm_in_2.x); 

  //Direction 2
  site_table[threadIdx.x] = tables[idx+2*size_dev];
  stag_phase              = phases[site_table[threadIdx.x]+2*size_dev];

  ferm_in_0 = tex1Dfetch(fermion_texRef,              site_table[threadIdx.x] + ferm_offset);
  ferm_in_1 = tex1Dfetch(fermion_texRef,   size_dev + site_table[threadIdx.x] + ferm_offset);
  ferm_in_2 = tex1Dfetch(fermion_texRef, 2*size_dev + site_table[threadIdx.x] + ferm_offset);
 
  LoadLinkRegs(gauge_texRef, size_dev, site_table[threadIdx.x]+gauge_offset, 2);

  ferm_out[0][0][threadIdx.x] -= link0.x*ferm_in_0.x+link0.y*ferm_in_0.y +
                                 link1.z*ferm_in_1.x+link1.w*ferm_in_1.y +
                                 stag_phase*(C1RE*ferm_in_2.x+ C1IM*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0.x*ferm_in_0.y-link0.y*ferm_in_0.x +
                                 link1.z*ferm_in_1.y-link1.w*ferm_in_1.x +
                                 stag_phase*(C1RE*ferm_in_2.y- C1IM*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0.z*ferm_in_0.x+link0.w*ferm_in_0.y +
                                 link2.x*ferm_in_1.x+link2.y*ferm_in_1.y +
                                 stag_phase*(C2RE*ferm_in_2.x+ C2IM*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0.z*ferm_in_0.y-link0.w*ferm_in_0.x +
                                 link2.x*ferm_in_1.y-link2.y*ferm_in_1.x +
                                 stag_phase*(C2RE*ferm_in_2.y- C2IM*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1.x*ferm_in_0.x+link1.y*ferm_in_0.y +
                                 link2.z*ferm_in_1.x+link2.w*ferm_in_1.y +
                                 stag_phase*(C3RE*ferm_in_2.x+ C3IM*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1.x*ferm_in_0.y-link1.y*ferm_in_0.x +
                                 link2.z*ferm_in_1.y-link2.w*ferm_in_1.x +
                                 stag_phase*(C3RE*ferm_in_2.y- C3IM*ferm_in_2.x); 

  //Direction 3
  site_table[threadIdx.x] = tables[idx+3*size_dev];
  stag_phase              = phases[site_table[threadIdx.x]+3*size_dev];

  ferm_in_0 = tex1Dfetch(fermion_texRef,              site_table[threadIdx.x] + ferm_offset);
  ferm_in_1 = tex1Dfetch(fermion_texRef,   size_dev + site_table[threadIdx.x] + ferm_offset);
  ferm_in_2 = tex1Dfetch(fermion_texRef, 2*size_dev + site_table[threadIdx.x] + ferm_offset);

  LoadLinkRegs(gauge_texRef, size_dev, site_table[threadIdx.x]+gauge_offset, 3);

  #ifndef IM_CHEM_POT
  ferm_out[0][0][threadIdx.x] -= link0.x*ferm_in_0.x+link0.y*ferm_in_0.y +
                                 link1.z*ferm_in_1.x+link1.w*ferm_in_1.y +
                                 stag_phase*(C1RE*ferm_in_2.x+  C1IM*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0.x*ferm_in_0.y-link0.y*ferm_in_0.x +
                                 link1.z*ferm_in_1.y-link1.w*ferm_in_1.x +
                                 stag_phase*(C1RE*ferm_in_2.y- C1IM*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0.z*ferm_in_0.x+link0.w*ferm_in_0.y +
                                 link2.x*ferm_in_1.x+link2.y*ferm_in_1.y +
                                 stag_phase*(C2RE*ferm_in_2.x+ C2IM*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0.z*ferm_in_0.y-link0.w*ferm_in_0.x +
                                 link2.x*ferm_in_1.y-link2.y*ferm_in_1.x +
                                 stag_phase*(C2RE*ferm_in_2.y- C2IM*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1.x*ferm_in_0.x+link1.y*ferm_in_0.y +
                                 link2.z*ferm_in_1.x+link2.w*ferm_in_1.y +
                                 stag_phase*(C3RE*ferm_in_2.x+ C3IM*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1.x*ferm_in_0.y-link1.y*ferm_in_0.x +
                                 link2.z*ferm_in_1.y-link2.w*ferm_in_1.x +
                                 stag_phase*(C3RE*ferm_in_2.y- C3IM*ferm_in_2.x); 
  #else
  ferm_aux_0.x = link0.x*ferm_in_0.x+link0.y*ferm_in_0.y +
                 link1.z*ferm_in_1.x+link1.w*ferm_in_1.y +
                 stag_phase*(C1RE*ferm_in_2.x+  C1IM*ferm_in_2.y); 

  ferm_aux_0.y = link0.x*ferm_in_0.y-link0.y*ferm_in_0.x +
                 link1.z*ferm_in_1.y-link1.w*ferm_in_1.x +
                 stag_phase*(C1RE*ferm_in_2.y- C1IM*ferm_in_2.x); 

  ferm_aux_1.x = link0.z*ferm_in_0.x+link0.w*ferm_in_0.y +
                 link2.x*ferm_in_1.x+link2.y*ferm_in_1.y +
                 stag_phase*(C2RE*ferm_in_2.x+ C2IM*ferm_in_2.y); 

  ferm_aux_1.y = link0.z*ferm_in_0.y-link0.w*ferm_in_0.x +
                 link2.x*ferm_in_1.y-link2.y*ferm_in_1.x +
                 stag_phase*(C2RE*ferm_in_2.y- C2IM*ferm_in_2.x); 

  ferm_aux_2.x = link1.x*ferm_in_0.x+link1.y*ferm_in_0.y +
                 link2.z*ferm_in_1.x+link2.w*ferm_in_1.y +
                 stag_phase*(C3RE*ferm_in_2.x+ C3IM*ferm_in_2.y); 

  ferm_aux_2.y = link1.x*ferm_in_0.y-link1.y*ferm_in_0.x +
                 link2.z*ferm_in_1.y-link2.w*ferm_in_1.x +
                 stag_phase*(C3RE*ferm_in_2.y- C3IM*ferm_in_2.x); 

  ferm_out[0][0][threadIdx.x] -=  ferm_aux_0.x*dev_eim_cos_f + ferm_aux_0.y*dev_eim_sin_f;  // Re[e^{-imu}*ferm_aux_0]
  ferm_out[0][1][threadIdx.x] -= -ferm_aux_0.x*dev_eim_sin_f + ferm_aux_0.y*dev_eim_cos_f;  // Im[e^{-imu}*ferm_aux_0]

  ferm_out[1][0][threadIdx.x] -=  ferm_aux_1.x*dev_eim_cos_f + ferm_aux_1.y*dev_eim_sin_f;  // Re[e^{-imu}*ferm_aux_1]
  ferm_out[1][1][threadIdx.x] -= -ferm_aux_1.x*dev_eim_sin_f + ferm_aux_1.y*dev_eim_cos_f;  // Im[e^{-imu}*ferm_aux_1]

  ferm_out[2][0][threadIdx.x] -=  ferm_aux_2.x*dev_eim_cos_f + ferm_aux_2.y*dev_eim_sin_f;  // Re[e^{-imu}*ferm_aux_2]
  ferm_out[2][1][threadIdx.x] -= -ferm_aux_2.x*dev_eim_sin_f + ferm_aux_2.y*dev_eim_cos_f;  // Im[e^{-imu}*ferm_aux_2]
  #endif

  //-------------------------------------------------end of second block
   
  out[idx               ].x = ferm_out[0][0][threadIdx.x]*(0.5f);
  out[idx               ].y = ferm_out[0][1][threadIdx.x]*(0.5f);
  out[idx +   size_dev  ].x = ferm_out[1][0][threadIdx.x]*(0.5f);
  out[idx +   size_dev  ].y = ferm_out[1][1][threadIdx.x]*(0.5f);
  out[idx + 2*size_dev  ].x = ferm_out[2][0][threadIdx.x]*(0.5f);
  out[idx + 2*size_dev  ].y = ferm_out[2][1][threadIdx.x]*(0.5f);
 
  //-------------------------------------------------end of DoeKernel
  }





///////////////////////////////////////////////////////////////////////////////////////// END OF KERNELS



void CuDoe(float2 *out, 
           float2 *in) 
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside CuDoe ...\033[0m\n");
  #endif

  dim3 BlockDimension(NUM_THREADS);
  dim3 GridDimension(size/BlockDimension.x);  //All sites 

  size_t gauge_field_size = sizeof(float4)*size*12;  
  size_t vector_size=3*size*sizeof(float2);

  size_t offset_g, offset_f;

  cudaSafe(AT,cudaBindTexture(&offset_g, gauge_texRef, gauge_field_device, gauge_field_size), "cudaBindTexture"); 
  offset_g/=sizeof(float4);

  cudaSafe(AT,cudaBindTexture(&offset_f, fermion_texRef, in, vector_size), "cudaBindTexture");
  offset_f/=sizeof(float2);

  DoeKernel<<<GridDimension,BlockDimension>>>(in, out, device_table, device_phases, offset_g, offset_f); 
  cudaCheckError(AT,"DoeKernel"); 

  cudaSafe(AT,cudaUnbindTexture(gauge_texRef), "cudaUnbindTexture");
  cudaSafe(AT,cudaUnbindTexture(fermion_texRef), "cudaUnbindTexture");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated CuDoe \033[0m\n");
  #endif
  }
