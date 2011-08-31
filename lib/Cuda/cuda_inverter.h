#ifndef CUDA_INVERTER_H
#define CUDA_INVERTER_H

extern "C" void cuda_shifted_inverter(const double residual, 
			              const double *shifts,
			              const int num_shifts,
                                      const int psferm,
			              int *ncount);

extern "C" void cuda_shifted_inverter_d(const double residual, 
			                const double *shifts,
  			                const int num_shifts,
                                        const int psferm,
			                int *ncount);

#endif
