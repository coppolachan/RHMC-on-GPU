#include<stdio.h>

extern "C" void GPUDeviceInfo(const int gpu_device) 
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

  printf("Selecting device %d\n",gpu_device);
  cudaSetDevice(gpu_device);
  }

