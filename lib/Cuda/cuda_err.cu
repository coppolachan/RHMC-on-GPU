#include "cuda_err.cuh"
#include <stdio.h>


void cudaSafe(const char* position, cudaError_t error, const char* message)
  {
  if(error!=cudaSuccess) 
    { 
    fprintf(stderr,"\033[31mERROR: %s : %s : %s\033[0m\n", position, 
	    message, cudaGetErrorString(error)); 
    exit(-1); 
    }
  }

void cudaCheckError(const char *position, const char *message)
     {
     cudaError_t error = cudaGetLastError();
     if(error!=cudaSuccess) 
       {
       fprintf(stderr,"\033[31mERROR: %s : %s : %s\033[0m\n", position, 
	       message, cudaGetErrorString(error)); 
       exit(-1);
       }
     } 


