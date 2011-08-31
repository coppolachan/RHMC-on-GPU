#include <stdio.h>

main(void)
  {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  if (deviceCount>0)  
    {
    printf("#########\n");
    for (device = 0; device < deviceCount; ++device) 
      {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      printf("Found CUDA device %d  : %s\n",device, deviceProp.name);
      printf("Compute capability   : %d.%d\n",deviceProp.major, deviceProp.minor);
      printf("#########\n");
      }
    } 
  else
    {
    printf("No CUDA device found \n");
    }
  }

