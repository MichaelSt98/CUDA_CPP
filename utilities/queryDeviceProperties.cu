#include <stdio.h> 
#include <iostream>




int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

  }
}

/**

See [cudaDeviceProp Struct Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)

int  ECCEnabled
int  accessPolicyMaxWindowSize
int  asyncEngineCount
int  canMapHostMemory
int  canUseHostPointerForRegisteredMem
int  clockRate
int  computeMode
int  computePreemptionSupported
int  concurrentKernels
int  concurrentManagedAccess
int  cooperativeLaunch
int  cooperativeMultiDeviceLaunch
int  deviceOverlap
int  directManagedMemAccessFromHost
int  globalL1CacheSupported
int  hostNativeAtomicSupported
int  integrated
int  isMultiGpuBoard
int  kernelExecTimeoutEnabled
int  l2CacheSize
int  localL1CacheSupported
char  luid[8]
unsigned int  luidDeviceNodeMask
int  major
int  managedMemory
int  maxBlocksPerMultiProcessor
int  maxGridSize[3]
int  maxSurface1D
int  maxSurface1DLayered[2]
int  maxSurface2D[2]
int  maxSurface2DLayered[3]
int  maxSurface3D[3]
int  maxSurfaceCubemap
int  maxSurfaceCubemapLayered[2]
int  maxTexture1D
int  maxTexture1DLayered[2]
int  maxTexture1DLinear
int  maxTexture1DMipmap
int  maxTexture2D[2]
int  maxTexture2DGather[2]
int  maxTexture2DLayered[3]
int  maxTexture2DLinear[3]
int  maxTexture2DMipmap[2]
int  maxTexture3D[3]
int  maxTexture3DAlt[3]
int  maxTextureCubemap
int  maxTextureCubemapLayered[2]
int  maxThreadsDim[3]
int  maxThreadsPerBlock
int  maxThreadsPerMultiProcessor
size_t  memPitch
int  memoryBusWidth
int  memoryClockRate
int  minor
int  multiGpuBoardGroupID
int  multiProcessorCount
char  name[256]
int  pageableMemoryAccess
int  pageableMemoryAccessUsesHostPageTables
int  pciBusID
int  pciDeviceID
int  pciDomainID
int  persistingL2CacheMaxSize
int  regsPerBlock
int  regsPerMultiprocessor
size_t  reservedSharedMemPerBlock
size_t  sharedMemPerBlock
size_t  sharedMemPerBlockOptin
size_t  sharedMemPerMultiprocessor
int  singleToDoublePrecisionPerfRatio
int  streamPrioritiesSupported
size_t  surfaceAlignment
int  tccDriver
size_t  textureAlignment
size_t  texturePitchAlignment
size_t  totalConstMem
size_t  totalGlobalMem
int  unifiedAddressing
cudaUUID_t  uuid
int  warpSize
*/

