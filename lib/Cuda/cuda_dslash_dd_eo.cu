// DOUBLE PRECISION KERNEL for even/odd fermions

__global__ void DslashDDKernelEO(double2  *out,
                                 double2  *in,
                                 int *tables, 
                                 int *phases, 
                                 size_t gauge_offset) 
  {
  int idx = blockIdx.x * blockDim.x + threadIdx.x + size_dev_h;  // idx>sizeh, ODD
  double stag_phase = 1.0;

  //Store result in sharedMem
  __shared__ double ferm_out[3][2][NUM_THREADS];

  //New tables indexing (index fastest)
  __shared__ int site_table[NUM_THREADS];

  //Load link matrix U_mu(ix) in registers
  double link0x, link0y, link0z, link0w, 
         link1x, link1y, link1z, link1w, 
         link2x, link2y, link2z, link2w;   
  float4 auxlink;

  double2 ferm_in_0, ferm_in_1, ferm_in_2;

  #ifdef IM_CHEM_POT
   double2 ferm_aux_0, ferm_aux_1, ferm_aux_2;
  #endif
 
  // DIRECTION 0
  site_table[threadIdx.x]  = tables[idx+4*size_dev];

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(0+3*0));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(1+3*0));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(2+3*0));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(0+3*0));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(1+3*0));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(2+3*0));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] = link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                                link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                                link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] = link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                                link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                                link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] = link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                                link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                                link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] = link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                                link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                                link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] = C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
                                C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
                                C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y; 
  ferm_out[2][1][threadIdx.x] = C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
                                C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
                                C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x; 

  //DIRECTION 1
  site_table[threadIdx.x] = tables[idx+5*size_dev];
  #ifdef USE_INTRINSIC 
  stag_phase              = __int2double_rn(phases[idx+size_dev]);
  #else
  stag_phase              = (double) phases[idx+size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(0+3*1));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(1+3*1));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(2+3*1));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(0+3*1));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(1+3*1));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(2+3*1));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] += link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                                 link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                                 link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                                 link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                                 link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                                 link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                                 link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                                 link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                                 link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
					     C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
					     C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
					     C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
					     C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x); 

  //DIRECTION 2
  site_table[threadIdx.x] = tables[idx+6*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[idx+2*size_dev]);
  #else
  stag_phase              = (double) phases[idx+2*size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(0+3*2));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(1+3*2));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(2+3*2));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(0+3*2));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(1+3*2));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(2+3*2));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] += link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                                 link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                                 link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                                 link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                                 link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                                 link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                                 link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                                 link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                                 link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
					     C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
					     C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
					     C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
					     C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x); 
  
  //DIRECTION 3
  site_table[threadIdx.x]  = tables[idx+7*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase               = __int2double_rn(phases[idx+3*size_dev]);
  #else
  stag_phase               = (double) phases[idx+3*size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(0+3*3));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(1+3*3));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(2+3*3));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(0+3*3));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(1+3*3));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(2+3*3));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  #ifndef IM_CHEM_POT
  ferm_out[0][0][threadIdx.x] += link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                                 link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                                 link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                                 link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                                 link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                                 link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                                 link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                                 link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                                 link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
					     C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
					     C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
					     C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
					     C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x); 
  #else
  ferm_aux_0.x = link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                 link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                 link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_aux_0.y = link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                 link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                 link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_aux_1.x = link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                 link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                 link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_aux_1.y = link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                 link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                 link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_aux_2.x = stag_phase*(C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
	                     C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
	                     C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y); 
  ferm_aux_2.y = stag_phase*(C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
	                     C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
	                     C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x); 

  ferm_out[0][0][threadIdx.x] += ferm_aux_0.x*dev_eim_cos_d - ferm_aux_0.y*dev_eim_sin_d;  // Re[e^{imu}*ferm_aux_0]
  ferm_out[0][1][threadIdx.x] += ferm_aux_0.x*dev_eim_sin_d + ferm_aux_0.y*dev_eim_cos_d;  // Im[e^{imu}*ferm_aux_0]

  ferm_out[1][0][threadIdx.x] += ferm_aux_1.x*dev_eim_cos_d - ferm_aux_1.y*dev_eim_sin_d;  // Re[e^{imu}*ferm_aux_1]
  ferm_out[1][1][threadIdx.x] += ferm_aux_1.x*dev_eim_sin_d + ferm_aux_1.y*dev_eim_cos_d;  // Im[e^{imu}*ferm_aux_1]

  ferm_out[2][0][threadIdx.x] += ferm_aux_2.x*dev_eim_cos_d - ferm_aux_2.y*dev_eim_sin_d;  // Re[e^{imu}*ferm_aux_2]
  ferm_out[2][1][threadIdx.x] += ferm_aux_2.x*dev_eim_sin_d + ferm_aux_2.y*dev_eim_cos_d;  // Im[e^{imu}*ferm_aux_2]
  #endif
  
  //---------------------------------------------------end of first block
 
  //DIRECTION 0
  site_table[threadIdx.x] = tables[idx];

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*0));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*0));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*0));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*0));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*0));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*0));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;
 
  ferm_out[0][0][threadIdx.x] -= link0x*ferm_in_0.x+link0y*ferm_in_0.y +
              			 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
				 C1RED*ferm_in_2.x   +C1IMD*ferm_in_2.y; 
  
  ferm_out[0][1][threadIdx.x] -= link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                                 C1RED*ferm_in_2.y   -C1IMD*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] -= link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                                 C2RED*ferm_in_2.x   +C2IMD*ferm_in_2.y; 

  ferm_out[1][1][threadIdx.x] -= link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                                 C2RED*ferm_in_2.y   -C2IMD*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] -= link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                                 C3RED*ferm_in_2.x   +C3IMD*ferm_in_2.y; 

  ferm_out[2][1][threadIdx.x] -= link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                                 C3RED*ferm_in_2.y   -C3IMD*ferm_in_2.x; 
  
  //DIRECTION 1
  site_table[threadIdx.x] = tables[idx+size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[site_table[threadIdx.x]+size_dev]);
  #else
  stag_phase              = (double) phases[site_table[threadIdx.x]+size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*1));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*1));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*1));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*1));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*1));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*1));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] -= link0x*ferm_in_0.x+link0y*ferm_in_0.y +
                                 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
                                 stag_phase*(C1RED*ferm_in_2.x+C1IMD*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                                 stag_phase*(C1RED*ferm_in_2.y-C1IMD*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                                 stag_phase*(C2RED*ferm_in_2.x+C2IMD*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                                 stag_phase*(C2RED*ferm_in_2.y-C2IMD*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                                 stag_phase*(C3RED*ferm_in_2.x+C3IMD*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                                 stag_phase*(C3RED*ferm_in_2.y- C3IMD*ferm_in_2.x); 

  //DIRECTION 2
  site_table[threadIdx.x] = tables[idx+2*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[site_table[threadIdx.x]+2*size_dev]);
  #else
  stag_phase              = (double) phases[site_table[threadIdx.x]+2*size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*2));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*2));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*2));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*2));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*2));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*2));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] -= link0x*ferm_in_0.x+link0y*ferm_in_0.y +
                                 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
                                 stag_phase*(C1RED*ferm_in_2.x+ C1IMD*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                                 stag_phase*(C1RED*ferm_in_2.y- C1IMD*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                                 stag_phase*(C2RED*ferm_in_2.x+ C2IMD*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                                 stag_phase*(C2RED*ferm_in_2.y- C2IMD*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                                 stag_phase*(C3RED*ferm_in_2.x+ C3IMD*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                                 stag_phase*(C3RED*ferm_in_2.y- C3IMD*ferm_in_2.x); 

  //DIRECTION 3
  site_table[threadIdx.x] = tables[idx+3*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[site_table[threadIdx.x]+3*size_dev]);
  #else
  stag_phase              = (double) phases[site_table[threadIdx.x]+3*size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*3));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*3));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*3));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*3));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*3));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*3));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  #ifndef IM_CHEM_POT
  ferm_out[0][0][threadIdx.x] -= link0x*ferm_in_0.x+link0y*ferm_in_0.y +
                                 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
                                 stag_phase*(C1RED*ferm_in_2.x+  C1IMD*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                                 stag_phase*(C1RED*ferm_in_2.y- C1IMD*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                                 stag_phase*(C2RED*ferm_in_2.x+ C2IMD*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                                 stag_phase*(C2RED*ferm_in_2.y- C2IMD*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                                 stag_phase*(C3RED*ferm_in_2.x+ C3IMD*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                                 stag_phase*(C3RED*ferm_in_2.y- C3IMD*ferm_in_2.x); 
  #else
  ferm_aux_0.x = link0x*ferm_in_0.x+link0y*ferm_in_0.y +
                 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
                 stag_phase*(C1RED*ferm_in_2.x+  C1IMD*ferm_in_2.y); 

  ferm_aux_0.y = link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                 stag_phase*(C1RED*ferm_in_2.y- C1IMD*ferm_in_2.x); 

  ferm_aux_1.x = link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                 stag_phase*(C2RED*ferm_in_2.x+ C2IMD*ferm_in_2.y); 

  ferm_aux_1.y = link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                 stag_phase*(C2RED*ferm_in_2.y- C2IMD*ferm_in_2.x); 

  ferm_aux_2.x = link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                 stag_phase*(C3RED*ferm_in_2.x+ C3IMD*ferm_in_2.y); 

  ferm_aux_2.y = link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                 stag_phase*(C3RED*ferm_in_2.y- C3IMD*ferm_in_2.x); 

  ferm_out[0][0][threadIdx.x] -=  ferm_aux_0.x*dev_eim_cos_d + ferm_aux_0.y*dev_eim_sin_d;  // Re[e^{-imu}*ferm_aux_0]
  ferm_out[0][1][threadIdx.x] -= -ferm_aux_0.x*dev_eim_sin_d + ferm_aux_0.y*dev_eim_cos_d;  // Im[e^{-imu}*ferm_aux_0]

  ferm_out[1][0][threadIdx.x] -=  ferm_aux_1.x*dev_eim_cos_d + ferm_aux_1.y*dev_eim_sin_d;  // Re[e^{-imu}*ferm_aux_1]
  ferm_out[1][1][threadIdx.x] -= -ferm_aux_1.x*dev_eim_sin_d + ferm_aux_1.y*dev_eim_cos_d;  // Im[e^{-imu}*ferm_aux_1]

  ferm_out[2][0][threadIdx.x] -=  ferm_aux_2.x*dev_eim_cos_d + ferm_aux_2.y*dev_eim_sin_d;  // Re[e^{-imu}*ferm_aux_2]
  ferm_out[2][1][threadIdx.x] -= -ferm_aux_2.x*dev_eim_sin_d + ferm_aux_2.y*dev_eim_cos_d;  // Im[e^{-imu}*ferm_aux_2]
  #endif

  //-------------------------------------------------end of second block

  // even
  ferm_in_0 = in[              idx - size_dev_h];
  ferm_in_1 = in[   size_dev + idx - size_dev_h];
  ferm_in_2 = in[ 2*size_dev + idx - size_dev_h];

  out[idx              - size_dev_h ].x = mass_d_dev*ferm_in_0.x;
  out[idx              - size_dev_h ].y = mass_d_dev*ferm_in_0.y;
  out[idx +   size_dev - size_dev_h ].x = mass_d_dev*ferm_in_1.x;
  out[idx +   size_dev - size_dev_h ].y = mass_d_dev*ferm_in_1.y;
  out[idx + 2*size_dev - size_dev_h ].x = mass_d_dev*ferm_in_2.x;
  out[idx + 2*size_dev - size_dev_h ].y = mass_d_dev*ferm_in_2.y;

  //odd
  out[idx               ].x = ferm_out[0][0][threadIdx.x]*(double)0.5;
  out[idx               ].y = ferm_out[0][1][threadIdx.x]*(double)0.5;
  out[idx +   size_dev  ].x = ferm_out[1][0][threadIdx.x]*(double)0.5;
  out[idx +   size_dev  ].y = ferm_out[1][1][threadIdx.x]*(double)0.5;
  out[idx + 2*size_dev  ].x = ferm_out[2][0][threadIdx.x]*(double)0.5;
  out[idx + 2*size_dev  ].y = ferm_out[2][1][threadIdx.x]*(double)0.5;

  //-------------------------------------------------end of Dslash
  }







__global__ void DslashDaggerDDKernelEO(double2 *out,
                                       double2 *in,
                                       int *tables, 
                                       int *phases,
                                       size_t gauge_offset) 
  { 
  int idx = blockIdx.x*blockDim.x + threadIdx.x;     // idx< sizeh, EVEN!!
  double stag_phase = 1.0;

  //Store result in sharedMem
  __shared__ double ferm_out[3][2][NUM_THREADS];

  #ifdef IM_CHEM_POT
   double2 ferm_aux_0, ferm_aux_1, ferm_aux_2;
  #endif

  //New tables indexing (index fastest)
  __shared__ int site_table[NUM_THREADS];

  //Load link matrix U_mu(ix) in registers
  double link0x, link0y, link0z, link0w, 
         link1x, link1y, link1z, link1w, 
         link2x, link2y, link2z, link2w;   
  float4 auxlink;

  double2 ferm_in_0, ferm_in_1, ferm_in_2;
  
  // DIRECTION 0
  site_table[threadIdx.x] = tables[idx+4*size_dev];

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];
 
  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(0+3*0));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(1+3*0));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(2+3*0));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(0+3*0));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(1+3*0));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(2+3*0));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] = link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                                link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                                link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] = link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                                link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                                link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] = link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                                link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                                link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] = link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                                link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                                link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] = C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
                                C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
                                C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y; 
  ferm_out[2][1][threadIdx.x] = C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
                                C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
                                C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x; 

  //DIRECTION 1
  site_table[threadIdx.x] = tables[idx+5*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[idx+size_dev]);
  #else
  stag_phase              = (double) phases[idx+size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(0+3*1));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(1+3*1));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(2+3*1));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(0+3*1));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(1+3*1));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(2+3*1));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] += link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                                 link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                                 link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                                 link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                                 link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                                 link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                                 link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                                 link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                                 link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
					     C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
					     C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
					     C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
					     C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x); 
   
  //DIRECTION 2
  site_table[threadIdx.x] = tables[idx+6*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[idx+2*size_dev]);
  #else
  stag_phase              = (double) phases[idx+2*size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(0+3*2));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(1+3*2));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(2+3*2));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(0+3*2));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(1+3*2));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(2+3*2));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] += link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                                 link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                                 link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                                 link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                                 link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                                 link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                                 link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                                 link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                                 link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
					     C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
					     C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
					     C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
					     C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x); 
  
  //DIRECTION 3
  site_table[threadIdx.x] = tables[idx+7*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[idx+3*size_dev]);
  #else
  stag_phase              = (double) phases[idx+3*size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(0+3*3));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(1+3*3));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, idx + gauge_offset + size_dev*(2+3*3));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(0+3*3));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(1+3*3));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + idx + gauge_offset + size_dev*(2+3*3));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  #ifndef IM_CHEM_POT
  ferm_out[0][0][threadIdx.x] += link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                                 link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                                 link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_out[0][1][threadIdx.x] += link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                                 link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                                 link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] += link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                                 link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                                 link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_out[1][1][threadIdx.x] += link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                                 link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                                 link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
					     C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
					     C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y); 
  ferm_out[2][1][threadIdx.x] += stag_phase*(C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
					     C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
					     C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x); 
  #else
  ferm_aux_0.x = link0x*ferm_in_0.x-link0y*ferm_in_0.y+  
                 link0z*ferm_in_1.x-link0w*ferm_in_1.y+ 
                 link1x*ferm_in_2.x-link1y*ferm_in_2.y; 
  ferm_aux_0.y = link0x*ferm_in_0.y+link0y*ferm_in_0.x+ 
                 link0z*ferm_in_1.y+link0w*ferm_in_1.x+ 
                 link1x*ferm_in_2.y+link1y*ferm_in_2.x; 

  ferm_aux_1.x = link1z*ferm_in_0.x-link1w*ferm_in_0.y+  
                 link2x*ferm_in_1.x-link2y*ferm_in_1.y+ 
                 link2z*ferm_in_2.x-link2w*ferm_in_2.y; 
  ferm_aux_1.y = link1z*ferm_in_0.y+link1w*ferm_in_0.x+ 
                 link2x*ferm_in_1.y+link2y*ferm_in_1.x+ 
                 link2z*ferm_in_2.y+link2w*ferm_in_2.x; 

  ferm_aux_2.x = stag_phase*(C1RED*ferm_in_0.x-C1IMD*ferm_in_0.y+  
	                     C2RED*ferm_in_1.x-C2IMD*ferm_in_1.y+ 
	                     C3RED*ferm_in_2.x-C3IMD*ferm_in_2.y); 
  ferm_aux_2.y = stag_phase*(C1RED*ferm_in_0.y+C1IMD*ferm_in_0.x+ 
	                     C2RED*ferm_in_1.y+C2IMD*ferm_in_1.x+ 
	                     C3RED*ferm_in_2.y+C3IMD*ferm_in_2.x); 

  ferm_out[0][0][threadIdx.x] += ferm_aux_0.x*dev_eim_cos_d - ferm_aux_0.y*dev_eim_sin_d;  // Re[e^{imu}*ferm_aux_0]
  ferm_out[0][1][threadIdx.x] += ferm_aux_0.x*dev_eim_sin_d + ferm_aux_0.y*dev_eim_cos_d;  // Im[e^{imu}*ferm_aux_0]

  ferm_out[1][0][threadIdx.x] += ferm_aux_1.x*dev_eim_cos_d - ferm_aux_1.y*dev_eim_sin_d;  // Re[e^{imu}*ferm_aux_1]
  ferm_out[1][1][threadIdx.x] += ferm_aux_1.x*dev_eim_sin_d + ferm_aux_1.y*dev_eim_cos_d;  // Im[e^{imu}*ferm_aux_1]

  ferm_out[2][0][threadIdx.x] += ferm_aux_2.x*dev_eim_cos_d - ferm_aux_2.y*dev_eim_sin_d;  // Re[e^{imu}*ferm_aux_2]
  ferm_out[2][1][threadIdx.x] += ferm_aux_2.x*dev_eim_sin_d + ferm_aux_2.y*dev_eim_cos_d;  // Im[e^{imu}*ferm_aux_2]
  #endif

  //---------------------------------------------------end of first block
 
  //DIRECTION 0
  site_table[threadIdx.x] = tables[idx];
 
  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*0));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*0));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*0));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*0));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*0));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*0));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] -= link0x*ferm_in_0.x+link0y*ferm_in_0.y +
              			 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
				 C1RED*ferm_in_2.x   +C1IMD*ferm_in_2.y; 
  
  ferm_out[0][1][threadIdx.x] -= link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                                 C1RED*ferm_in_2.y   -C1IMD*ferm_in_2.x; 

  ferm_out[1][0][threadIdx.x] -= link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                                 C2RED*ferm_in_2.x   +C2IMD*ferm_in_2.y; 

  ferm_out[1][1][threadIdx.x] -= link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                                 C2RED*ferm_in_2.y   -C2IMD*ferm_in_2.x; 

  ferm_out[2][0][threadIdx.x] -= link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                                 C3RED*ferm_in_2.x   +C3IMD*ferm_in_2.y; 

  ferm_out[2][1][threadIdx.x] -= link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                                 C3RED*ferm_in_2.y   -C3IMD*ferm_in_2.x; 
  
  //DIRECTION 1
  site_table[threadIdx.x] = tables[idx+size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[site_table[threadIdx.x]+size_dev]);
  #else
  stag_phase              = (double) phases[site_table[threadIdx.x]+size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float 
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*1));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*1));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*1));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*1));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*1));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*1));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] -= link0x*ferm_in_0.x+link0y*ferm_in_0.y +
                                 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
                                 stag_phase*(C1RED*ferm_in_2.x+C1IMD*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                                 stag_phase*(C1RED*ferm_in_2.y-C1IMD*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                                 stag_phase*(C2RED*ferm_in_2.x+C2IMD*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                                 stag_phase*(C2RED*ferm_in_2.y-C2IMD*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                                 stag_phase*(C3RED*ferm_in_2.x+C3IMD*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                                 stag_phase*(C3RED*ferm_in_2.y- C3IMD*ferm_in_2.x); 

  //DIRECTION 2
  site_table[threadIdx.x] = tables[idx+2*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[site_table[threadIdx.x]+2*size_dev]);
  #else
  stag_phase              = (double) phases[site_table[threadIdx.x]+2*size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*2));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*2));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*2));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*2));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*2));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*2));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  ferm_out[0][0][threadIdx.x] -= link0x*ferm_in_0.x+link0y*ferm_in_0.y +
                                 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
                                 stag_phase*(C1RED*ferm_in_2.x+ C1IMD*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                                 stag_phase*(C1RED*ferm_in_2.y- C1IMD*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                                 stag_phase*(C2RED*ferm_in_2.x+ C2IMD*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                                 stag_phase*(C2RED*ferm_in_2.y- C2IMD*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                                 stag_phase*(C3RED*ferm_in_2.x+ C3IMD*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                                 stag_phase*(C3RED*ferm_in_2.y- C3IMD*ferm_in_2.x); 

  //DIRECTION 3
  site_table[threadIdx.x] = tables[idx+3*size_dev];
  #ifdef USE_INTRINSIC
  stag_phase              = __int2double_rn(phases[site_table[threadIdx.x]+3*size_dev]);
  #else
  stag_phase              = (double) phases[site_table[threadIdx.x]+3*size_dev];
  #endif

  ferm_in_0 = in[              site_table[threadIdx.x]];
  ferm_in_1 = in[   size_dev + site_table[threadIdx.x]];
  ferm_in_2 = in[ 2*size_dev + site_table[threadIdx.x]];

  // 1st float
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*3));
  link0x=(double) auxlink.x;
  link0y=(double) auxlink.y;
  link0z=(double) auxlink.z;
  link0w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*3));
  link1x=(double) auxlink.x;
  link1y=(double) auxlink.y;
  link1z=(double) auxlink.z;
  link1w=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*3));
  link2x=(double) auxlink.x;
  link2y=(double) auxlink.y;
  link2z=(double) auxlink.z;
  link2w=(double) auxlink.w;
  // 2nd float
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(0+3*3));
  link0x+=(double) auxlink.x;
  link0y+=(double) auxlink.y;
  link0z+=(double) auxlink.z;
  link0w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(1+3*3));
  link1x+=(double) auxlink.x;
  link1y+=(double) auxlink.y;
  link1z+=(double) auxlink.z;
  link1w+=(double) auxlink.w;
  auxlink = tex1Dfetch(gauge_texRef, 12*size_dev + site_table[threadIdx.x] + gauge_offset + size_dev*(2+3*3));
  link2x+=(double) auxlink.x;
  link2y+=(double) auxlink.y;
  link2z+=(double) auxlink.z;
  link2w+=(double) auxlink.w;

  #ifndef IM_CHEM_POT
  ferm_out[0][0][threadIdx.x] -= link0x*ferm_in_0.x+link0y*ferm_in_0.y +
                                 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
                                 stag_phase*(C1RED*ferm_in_2.x+  C1IMD*ferm_in_2.y); 

  ferm_out[0][1][threadIdx.x] -= link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                                 stag_phase*(C1RED*ferm_in_2.y- C1IMD*ferm_in_2.x); 

  ferm_out[1][0][threadIdx.x] -= link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                                 stag_phase*(C2RED*ferm_in_2.x+ C2IMD*ferm_in_2.y); 

  ferm_out[1][1][threadIdx.x] -= link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                                 stag_phase*(C2RED*ferm_in_2.y- C2IMD*ferm_in_2.x); 

  ferm_out[2][0][threadIdx.x] -= link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                                 stag_phase*(C3RED*ferm_in_2.x+ C3IMD*ferm_in_2.y); 

  ferm_out[2][1][threadIdx.x] -= link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                                 stag_phase*(C3RED*ferm_in_2.y- C3IMD*ferm_in_2.x); 
  #else
  ferm_aux_0.x = link0x*ferm_in_0.x+link0y*ferm_in_0.y +
                 link1z*ferm_in_1.x+link1w*ferm_in_1.y +
                 stag_phase*(C1RED*ferm_in_2.x+  C1IMD*ferm_in_2.y); 

  ferm_aux_0.y = link0x*ferm_in_0.y-link0y*ferm_in_0.x +
                 link1z*ferm_in_1.y-link1w*ferm_in_1.x +
                 stag_phase*(C1RED*ferm_in_2.y- C1IMD*ferm_in_2.x); 

  ferm_aux_1.x = link0z*ferm_in_0.x+link0w*ferm_in_0.y +
                 link2x*ferm_in_1.x+link2y*ferm_in_1.y +
                 stag_phase*(C2RED*ferm_in_2.x+ C2IMD*ferm_in_2.y); 

  ferm_aux_1.y = link0z*ferm_in_0.y-link0w*ferm_in_0.x +
                 link2x*ferm_in_1.y-link2y*ferm_in_1.x +
                 stag_phase*(C2RED*ferm_in_2.y- C2IMD*ferm_in_2.x); 

  ferm_aux_2.x = link1x*ferm_in_0.x+link1y*ferm_in_0.y +
                 link2z*ferm_in_1.x+link2w*ferm_in_1.y +
                 stag_phase*(C3RED*ferm_in_2.x+ C3IMD*ferm_in_2.y); 

  ferm_aux_2.y = link1x*ferm_in_0.y-link1y*ferm_in_0.x +
                 link2z*ferm_in_1.y-link2w*ferm_in_1.x +
                 stag_phase*(C3RED*ferm_in_2.y- C3IMD*ferm_in_2.x); 

  ferm_out[0][0][threadIdx.x] -=  ferm_aux_0.x*dev_eim_cos_d + ferm_aux_0.y*dev_eim_sin_d;  // Re[e^{-imu}*ferm_aux_0]
  ferm_out[0][1][threadIdx.x] -= -ferm_aux_0.x*dev_eim_sin_d + ferm_aux_0.y*dev_eim_cos_d;  // Im[e^{-imu}*ferm_aux_0]

  ferm_out[1][0][threadIdx.x] -=  ferm_aux_1.x*dev_eim_cos_d + ferm_aux_1.y*dev_eim_sin_d;  // Re[e^{-imu}*ferm_aux_1]
  ferm_out[1][1][threadIdx.x] -= -ferm_aux_1.x*dev_eim_sin_d + ferm_aux_1.y*dev_eim_cos_d;  // Im[e^{-imu}*ferm_aux_1]

  ferm_out[2][0][threadIdx.x] -=  ferm_aux_2.x*dev_eim_cos_d + ferm_aux_2.y*dev_eim_sin_d;  // Re[e^{-imu}*ferm_aux_2]
  ferm_out[2][1][threadIdx.x] -= -ferm_aux_2.x*dev_eim_sin_d + ferm_aux_2.y*dev_eim_cos_d;  // Im[e^{-imu}*ferm_aux_2]
  #endif

  //-------------------------------------------------end of second block

  // even   
  ferm_in_0 = in[              idx];
  ferm_in_1 = in[   size_dev + idx];
  ferm_in_2 = in[ 2*size_dev + idx];

  out[idx               ].x = mass_d_dev*ferm_in_0.x - ferm_out[0][0][threadIdx.x]*(double)0.5;
  out[idx               ].y = mass_d_dev*ferm_in_0.y - ferm_out[0][1][threadIdx.x]*(double)0.5;
  out[idx +   size_dev  ].x = mass_d_dev*ferm_in_1.x - ferm_out[1][0][threadIdx.x]*(double)0.5;
  out[idx +   size_dev  ].y = mass_d_dev*ferm_in_1.y - ferm_out[1][1][threadIdx.x]*(double)0.5;
  out[idx + 2*size_dev  ].x = mass_d_dev*ferm_in_2.x - ferm_out[2][0][threadIdx.x]*(double)0.5;
  out[idx + 2*size_dev  ].y = mass_d_dev*ferm_in_2.y - ferm_out[2][1][threadIdx.x]*(double)0.5;

  // odd
  out[idx              + size_dev_h ].x = (double)0.0;
  out[idx              + size_dev_h ].y = (double)0.0;
  out[idx +   size_dev + size_dev_h ].x = (double)0.0;
  out[idx +   size_dev + size_dev_h ].y = (double)0.0;
  out[idx + 2*size_dev + size_dev_h ].x = (double)0.0;
  out[idx + 2*size_dev + size_dev_h ].y = (double)0.0;

  //-------------------------------------------------end of DslashDagger
  }






/*
================================================================= EXTERNAL C FUNCTION
*/

void DslashOperatorDDEO(double2 *out, 
 		        double2 *in, 
 		        const int isign)
  {
  #ifdef DEBUG_MODE_2
  printf("\033[32mDEBUG: inside DslashOperatorDDEO ...\033[0m\n");
  #endif

  dim3 BlockDimension(NUM_THREADS);
  dim3 GridDimension(sizeh/BlockDimension.x);  //Half sites

  size_t gauge_field_size = sizeof(float4)*size*12;  

  size_t offset_g;
  cudaSafe(AT,cudaBindTexture(&offset_g, gauge_texRef, gauge_field_device, 2*gauge_field_size), "cudaBindTexture");  
  offset_g/=sizeof(float4);

  if(isign == PLUS) 
    {
    DslashDDKernelEO<<<GridDimension,BlockDimension>>>(out, in, device_table, device_phases, offset_g); 
    cudaCheckError(AT,"DslashDDKernelEO"); 
    }
  
  if(isign == MINUS) 
    {
    DslashDaggerDDKernelEO<<<GridDimension,BlockDimension>>>(out, in, device_table, device_phases, offset_g); 
    cudaCheckError(AT,"DslashDaggerDDKernelEO"); 
    }

  cudaSafe(AT,cudaUnbindTexture(gauge_texRef), "cudaUnbindTexture");

  #ifdef DEBUG_MODE_2
  printf("\033[32m\tterminated DslashOperatorDDEO \033[0m\n");
  #endif
  }

