#ifndef CUDA_INIT_H
#define CUDA_INIT_H

extern "C" void cuda_init0(void);
extern "C" void cuda_init1(void);
extern "C" void cuda_meas_init(void);
extern "C" void cuda_meas_end(void);
extern "C" void cuda_end(void);
extern "C" void cuda_get_conf(void);
extern "C" void cuda_get_momenta(void);
extern "C" void cuda_put_momenta(void);

#endif
