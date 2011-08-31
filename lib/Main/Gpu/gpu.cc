#ifdef USE_GPU
#include "include/global_var.h"
#include "include/gpu.h"


#include"lib/Cuda/cuda_init.h"
#include"lib/Tools/Packer/packer.h"


void gpu_init0(void)
  {
  smartpack_gauge(gauge_field_packed, gauge_conf);
  make_shift_table(shift_table);
  cuda_init0(); 
  }

void gpu_init1(void)
  {
  smartpack_multifermion(chi_packed, fermion_phi);
  smartpack_tamatrix(ipdot_packed, gauge_ipdot);
  smartpack_thmatrix(momenta_packed, gauge_momenta);
  cuda_init1(); 
  }

void gpu_end(void)
  {
  cuda_end();
  smartunpack_gauge(gauge_conf, gauge_field_packed);
  smartunpack_tamatrix(gauge_ipdot, ipdot_packed);
  smartunpack_thmatrix(gauge_momenta, momenta_packed);
  }

#endif
