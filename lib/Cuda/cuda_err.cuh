#ifndef CUDA_ERR_H_
#define CUDA_ERR_H_

void cudaSafe(const char* position, cudaError_t error, const char* message);

void cudaCheckError(const char *position, const char *message);

#endif 