include <stdio.h>
 
// Print device properties
void printDevProp(cudaDeviceProp devProp, int amount_of_info = 0)
{
    printf("---------------------------------------------------------------")
    if (amount_of_info >= 0) {
        printf("Name:                          \t%s\n", devProp.name);
        printf("Major revision number:         \t%d\n", devProp.major);
        printf("Minor revision number:         \t%d\n", devProp.minor);
    }
    if (amount_of_info >= 1) {
        printf("Total global memory:           \t%u\n", devProp.totalGlobalMem);
        printf("Total shared memory per block: \t%u\n", devProp.sharedMemPerBlock);
        printf("Total registers per block:     \t%d\n", devProp.regsPerBlock);
        printf("Warp size:                     \t%d\n", devProp.warpSize);
        printf("Clock rate:                    \t%d\n", devProp.clockRate);
        printf("Total constant memory:         \t%u\n", devProp.totalConstMem);
    }
    if (amount_of_info >= 2) {
        printf("Maximum memory pitch:          \t%u\n", devProp.memPitch);
        printf("Maximum threads per block:     \t%d\n", devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i) {
            printf("Maximum dimension %d of block:  \t%d\n", i, devProp.maxThreadsDim[i]);
        }
        for (int i = 0; i < 3; ++i) {
            printf("Maximum dimension %d of grid:   \t%d\n", i, devProp.maxGridSize[i]);
        }
        printf("Texture alignment:             \t%u\n", devProp.textureAlignment);
        printf("Concurrent copy and execution: \t%s\n", (devProp.deviceOverlap ? "Yes" : "No"));
        printf("Number of multiprocessors:     \t%d\n", devProp.multiProcessorCount);
    }
    if (amount_of_info >= 3) {
        printf("Kernel execution timeout:      \t%s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        printf("ECCEnabled:                    \t%s\n", devProp.ECCEnabled ? "Yes" : "No");
        printf("accessPolicyMaxWindowSize:     \t%d\n", devProp.accessPolicyMaxWindowSize);
        printf("asyncEngineCount:              \t%d\n", devProp.asyncEngineCount);
        printf("canMapHostMemory:              \t%d\n", devProp.canMapHostMemory);
        printf("canUseHostPointerForRegMem:    \t%d\n", devProp.canUseHostPointerForRegisteredMem);
        printf("computeMode:                   \t%d\n", devProp.computeMode);
        printf("computePreemptionSupported:    \t%d\n", devProp.computePreemptionSupported);
        printf("concurrentKernels:             \t%d\n", devProp.concurrentKernels);
        printf("concurrentManagedAccess:       \t%d\n", devProp.concurrentManagedAccess);
        printf("cooperativeLaunch:             \t%d\n", devProp.cooperativeLaunch);
        printf("cooperativeMultiDeviceLaunch:  \t%d\n", devProp.cooperativeMultiDeviceLaunch);
        printf("deviceOverlap:                 \t%d\n", devProp.deviceOverlap);
        printf("directManagedMemAccessFromHost:\t%d\n", devProp.directManagedMemAccessFromHost);
        printf("globalL1CacheSupported:        \t%d\n", devProp.globalL1CacheSupported);
        printf("hostNativeAtomicSupported:     \t%d\n", devProp.hostNativeAtomicSupported);
        printf("integrated:                    \t%d\n", devProp.integrated);
        printf("isMultiGpuBoard:               \t%d\n", devProp.isMultiGpuBoard);
        printf("kernelExecTimeoutEnabled:      \t%d\n", devProp.kernelExecTimeoutEnabled);
        printf("l2CacheSize:                   \t%d\n", devProp.l2CacheSize);
        printf("localL1CacheSupported:         \t%d\n", devProp.localL1CacheSupported);
        printf("luid:                          \t%s\n", devProp.luid);
        printf("luidDeviceNodeMask:            \t%d\n", devProp.luidDeviceNodeMask);
        printf("managedMemory                  \t%d\n", devProp.managedMemory);
        printf("maxBlocksPerMultiProcessor:    \t%d\n", devProp.maxBlocksPerMultiProcessor);
        for (int i = 0; i < 3; ++i) {
            printf("maxGridSize dim       %d:      \t%d\n", i, devProp.maxGridSize[i]);
        }
        printf("maxSurface1D                   \t%d\n", devProp.maxSurface1D);
        for (int i = 0; i < 2; ++i) {
            printf("maxSurface1DLayered dim %d:    \t%d\n", i, devProp.maxSurface1DLayered[i]);
        }
        for (int i = 0; i < 2; ++i) {
            printf("maxSurface2D dim %d:           \t%d\n", i, devProp.maxSurface2D[i]);
        }
        for (int i = 0; i < 3; ++i) {
            printf("maxSurface2DLayered dim %d:    \t%d\n", i, devProp.maxSurface2DLayered[i]);
        }
        for (int i = 0; i < 3; ++i) {
            printf("maxSurface3D dim %d:           \t%d\n", i, devProp.maxSurface3D[i]);
        }
        printf("maxSurfaceCubemap:             \t%d\n", devProp.maxSurfaceCubemap);
        for (int i = 0; i < 2; ++i) {
            printf("maxSurfaceCubemapLayered dim %d:\t%d\n", i, devProp.maxSurfaceCubemapLayered[i]);
        }
        printf("maxTexture1D                   \t%d\n", devProp.maxTexture1D);
        for (int i = 0; i < 2; ++i) {
            printf("maxTexture1DLayered dim %d:      \t%d\n", i, devProp.maxTexture1DLayered[i]);
        }
        printf("maxTexture1DLinear:            \t%d\n", devProp.maxTexture1DLinear);
        printf("maxTexture1DMipmap:            \t%d\n", devProp.maxTexture1DMipmap);
        for (int i = 0; i < 2; ++i) {
            printf("maxTexture2D dim %d:             \t%d\n", i, devProp.maxTexture2D[i]);
        }
        for (int i = 0; i < 2; ++i) {
            printf("maxTexture2DGather dim %d:        \t%d\n", i, devProp.maxTexture2DGather[i]);
        }
        for (int i = 0; i < 3; ++i) {
            printf("maxTexture2DLayered dim %d:       \t%d\n", i, devProp.maxTexture2DLayered[i]);
        }
        for (int i = 0; i < 3; ++i) {
            printf("maxTexture2DLinear dim %d:        \t%d\n", i, devProp.maxTexture2DLinear[i]);
        }
        for (int i = 0; i < 2; ++i) {
            printf("maxTexture2DMipmap dim %d:        \t%d\n", i, devProp.maxTexture2DMipmap[i]);
        }
        for (int i = 0; i < 3; ++i) {
            printf("maxTexture3D dim %d:              \t%d\n", i, devProp.maxTexture3D[i]);
        }
        for (int i = 0; i < 3; ++i) {
            printf("maxTexture3DAlt dim %d:           \t%d\n", i, devProp.maxTexture3DAlt[i]);
        }
        printf("maxTextureCubemap:                \t%d\n", devProp.maxTextureCubemap);
        for (int i = 0; i < 2; ++i) {
            printf("maxTextureCubemapLayered dim %d:  \t%d\n", i, devProp.maxTextureCubemapLayered[i]);
        }
        for (int i = 0; i < 3; ++i) {
            printf("maxThreadsDim dim %d:             \t%d\n", i, devProp.maxThreadsDim[i]);
        }
        printf("maxThreadsPerBlock:               \t%d\n", devProp.maxThreadsPerBlock);
        printf("maxThreadsPerMultiProcessor:      \t%d\n", devProp.maxThreadsPerMultiProcessor);
        printf("memPitch:                         \t%zu\n", devProp.memPitch);
        printf("memoryBusWidth:                   \t%d\n", devProp.memoryBusWidth);
        printf("memoryClockRate:                  \t%d\n", devProp.memoryClockRate);
        printf("multiGpuBoardGroupID:             \t%d\n", devProp.multiGpuBoardGroupID);
        printf("multiProcessorCount:              \t%d\n", devProp.multiProcessorCount);
        printf("pageableMemoryAccess:             \t%d\n", devProp.pageableMemoryAccess);
        printf("pageableMemAccessUsesHost:        \t%d\n", devProp.pageableMemoryAccessUsesHostPageTables);
        printf("pciBusID:                         \t%d\n", devProp.pciBusID);
        printf("pciDeviceID:                      \t%d\n", devProp.pciDeviceID);
        printf("pciDomainID:                      \t%d\n", devProp.pciDomainID);
        printf("persistingL2CacheMaxSize:         \t%d\n", devProp.persistingL2CacheMaxSize);
        printf("regsPerBlock:                     \t%d\n", devProp.regsPerBlock);
        printf("regsPerMultiprocessor:            \t%d\n", devProp.regsPerMultiprocessor);
        printf("reservedSharedMemPerBlock:        \t%zu\n", devProp.reservedSharedMemPerBlock);
        printf("sharedMemPerBlock:                \t%zu\n", devProp.sharedMemPerBlock);
        printf("sharedMemPerBlockOptin:           \t%zu\n", devProp.sharedMemPerBlockOptin);
        printf("sharedMemPerMultiprocessor:       \t%zu\n", devProp.sharedMemPerMultiprocessor);
        printf("singleToDoublePrecisionPerfRatio: \t%zu\n", devProp.singleToDoublePrecisionPerfRatio);
        printf("streamPrioritiesSupported:        \t%d\n", devProp.streamPrioritiesSupported);
        printf("surfaceAlignment:                 \t%zu\n", devProp.surfaceAlignment);
        printf("tccDriver:                        \t%d\n", devProp.tccDriver);
        printf("textureAlignment:                 \t%zu\n", devProp.textureAlignment);
        printf("texturePitchAlignment:            \t%zu\n", devProp.texturePitchAlignment);
        printf("totalConstMem:                    \t%zu\n", devProp.totalConstMem);
        printf("totalGlobalMem:                   \t%zu\n", devProp.totalGlobalMem);
        printf("unifiedAddressing:                \t%d\n", devProp.unifiedAddressing);

        %cudaUUID_t  uuid
    }
    printf("---------------------------------------------------------------")
}
 
int main()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
 
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device number: %d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
 
    //printf("\nPress any key to exit...");
    //char c;
    //scanf("%c", &c);
 
    return 0;
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


