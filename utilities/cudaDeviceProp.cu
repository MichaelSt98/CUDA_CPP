#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

template<typename T_1, typename T_2> void print(int length, T_1 description, T_2 value, int dim=-1) {
    std::ostringstream s_description;
    if (dim == -1) {
        s_description << description << ":";
    }
    else {
        s_description << description << " (dim = " << dim << "):";
    }
    std::cout << std::left;
    std::cout << std::setw(length) << s_description.str();
    std::cout << value;
    std::cout << std::endl;
}

 
// Print device properties
void printDevProp(cudaDeviceProp devProp, int amount_of_info = 0, int length = 40)
{
    std::cout << "---------------------------------------------------------------" << std::endl;
    if (amount_of_info >= 0) {
        print(length, "Name", devProp.name);
        print(length, "Major revision number", devProp.major);
        print(length, "Minor revision number", devProp.minor);
    }
    if (amount_of_info >= 1) {
        print(length, "Total global memory", devProp.totalGlobalMem);
        print(length, "Total shared memory per block", devProp.sharedMemPerBlock);
        print(length, "Total registers per block", devProp.regsPerBlock);
        print(length, "Warp size", devProp.warpSize);
        print(length, "Clock rate", devProp.clockRate);
        print(length, "Total constant memory", devProp.totalConstMem);
    }
    if (amount_of_info >= 2) {
        print(length, "Maximum memory pitch", devProp.memPitch);
        print(length, "Maximum threads per block", devProp.maxThreadsPerBlock);
        for (int i = 0; i < 3; ++i) {
            print(length, "Maximum dimension of block", devProp.maxThreadsDim[i], i);
        }
        for (int i = 0; i < 3; ++i) {
            print(length, "Maximum dimension of grid:", devProp.maxGridSize[i], i);
        }
        print(length, "Texture alignment", devProp.textureAlignment);
        print(length, "Concurrent copy and execution", (devProp.deviceOverlap ? "Yes" : "No"));
        print(length, "Number of multiprocessors", devProp.multiProcessorCount);
    }
    if (amount_of_info >= 3) {
        print(length, "Kernel execution timeout", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
        print(length, "ECCEnabled:", devProp.ECCEnabled ? "Yes" : "No");
        //print(length, "accessPolicyMaxWindowSize:     \t%d\n", devProp.accessPolicyMaxWindowSize);
        print(length, "asyncEngineCount", devProp.asyncEngineCount);
        print(length, "canMapHostMemory", devProp.canMapHostMemory);
        print(length, "canUseHostPointerForRegMem", devProp.canUseHostPointerForRegisteredMem);
        print(length, "computeMode", devProp.computeMode);
        print(length, "computePreemptionSupported", devProp.computePreemptionSupported);
        print(length, "concurrentKernels", devProp.concurrentKernels);
        print(length, "concurrentManagedAccess", devProp.concurrentManagedAccess);
        print(length, "cooperativeLaunch", devProp.cooperativeLaunch);
        print(length, "cooperativeMultiDeviceLaunch", devProp.cooperativeMultiDeviceLaunch);
        print(length, "deviceOverlap", devProp.deviceOverlap);
        print(length, "directManagedMemAccessFromHost", devProp.directManagedMemAccessFromHost);
        print(length, "globalL1CacheSupported", devProp.globalL1CacheSupported);
        print(length, "hostNativeAtomicSupported", devProp.hostNativeAtomicSupported);
        print(length, "integrated", devProp.integrated);
        print(length, "isMultiGpuBoard", devProp.isMultiGpuBoard);
        print(length, "kernelExecTimeoutEnabled", devProp.kernelExecTimeoutEnabled);
        print(length, "l2CacheSize", devProp.l2CacheSize);
        print(length, "localL1CacheSupported", devProp.localL1CacheSupported);
        print(length, "luid", devProp.luid);
        print(length, "luidDeviceNodeMask", devProp.luidDeviceNodeMask);
        print(length, "managedMemory", devProp.managedMemory);
        //print(length, "maxBlocksPerMultiProcessor:    \t%d\n", devProp.maxBlocksPerMultiProcessor);
        for (int i = 0; i < 3; ++i) {
            print(length, "maxGridSize", devProp.maxGridSize[i], i);
        }
        print(length, "maxSurface1D", devProp.maxSurface1D);
        for (int i = 0; i < 2; ++i) {
            print(length, "maxSurface1DLayered", devProp.maxSurface1DLayered[i], i);
        }
        for (int i = 0; i < 2; ++i) {
            print(length, "maxSurface2D", devProp.maxSurface2D[i], i);
        }
        for (int i = 0; i < 3; ++i) {
            print(length, "maxSurface2DLayered", devProp.maxSurface2DLayered[i], i);
        }
        for (int i = 0; i < 3; ++i) {
            print(length, "maxSurface3D", devProp.maxSurface3D[i], i);
        }
        print(length, "maxSurfaceCubemap", devProp.maxSurfaceCubemap);
        for (int i = 0; i < 2; ++i) {
            print(length, "maxSurfaceCubemapLayered", devProp.maxSurfaceCubemapLayered[i], i);
        }
        print(length, "maxTexture1D", devProp.maxTexture1D);
        for (int i = 0; i < 2; ++i) {
            print(length, "maxTexture1DLayered", devProp.maxTexture1DLayered[i], i);
        }
        print(length, "maxTexture1DLinear", devProp.maxTexture1DLinear);
        print(length, "maxTexture1DMipmap", devProp.maxTexture1DMipmap);
        for (int i = 0; i < 2; ++i) {
            print(length, "maxTexture2D", devProp.maxTexture2D[i], i);
        }
        for (int i = 0; i < 2; ++i) {
            print(length, "maxTexture2DGather", devProp.maxTexture2DGather[i], i);
        }
        for (int i = 0; i < 3; ++i) {
            print(length, "maxTexture2DLayered", devProp.maxTexture2DLayered[i], i);
        }
        for (int i = 0; i < 3; ++i) {
            print(length, "maxTexture2DLinear", devProp.maxTexture2DLinear[i], i);
        }
        for (int i = 0; i < 2; ++i) {
            print(length, "maxTexture2DMipmap", devProp.maxTexture2DMipmap[i], i);
        }
        for (int i = 0; i < 3; ++i) {
            print(length, "maxTexture3D", devProp.maxTexture3D[i], i);
        }
        for (int i = 0; i < 3; ++i) {
            print(length, "maxTexture3DAlt", devProp.maxTexture3DAlt[i], i);
        }
        print(length, "maxTextureCubemap", devProp.maxTextureCubemap);
        for (int i = 0; i < 2; ++i) {
            print(length, "maxTextureCubemapLayered", devProp.maxTextureCubemapLayered[i], i);
        }
        for (int i = 0; i < 3; ++i) {
            print(length, "maxThreadsDim", devProp.maxThreadsDim[i], i);
        }
        print(length, "maxThreadsPerBlock", devProp.maxThreadsPerBlock);
        print(length, "maxThreadsPerMultiProcessor", devProp.maxThreadsPerMultiProcessor);
        print(length, "memPitch", devProp.memPitch);
        print(length, "memoryBusWidth", devProp.memoryBusWidth);
        print(length, "memoryClockRate", devProp.memoryClockRate);
        print(length, "multiGpuBoardGroupID", devProp.multiGpuBoardGroupID);
        print(length, "multiProcessorCount", devProp.multiProcessorCount);
        print(length, "pageableMemoryAccess", devProp.pageableMemoryAccess);
        print(length, "pageableMemAccessUsesHost", devProp.pageableMemoryAccessUsesHostPageTables);
        print(length, "pciBusID", devProp.pciBusID);
        print(length, "pciDeviceID", devProp.pciDeviceID);
        print(length, "pciDomainID", devProp.pciDomainID);
        //print(length, "persistingL2CacheMaxSize:         \t%d\n", devProp.persistingL2CacheMaxSize);
        print(length, "regsPerBlock", devProp.regsPerBlock);
        print(length, "regsPerMultiprocessor", devProp.regsPerMultiprocessor);
        //print(length, "reservedSharedMemPerBlock:        \t%zu\n", devProp.reservedSharedMemPerBlock);
        print(length, "sharedMemPerBlock", devProp.sharedMemPerBlock);
        print(length, "sharedMemPerBlockOptin", devProp.sharedMemPerBlockOptin);
        print(length, "sharedMemPerMultiprocessor", devProp.sharedMemPerMultiprocessor);
        print(length, "singleToDoublePrecisionPerfRatio", devProp.singleToDoublePrecisionPerfRatio);
        print(length, "streamPrioritiesSupported", devProp.streamPrioritiesSupported);
        print(length, "surfaceAlignment", devProp.surfaceAlignment);
        print(length, "tccDriver", devProp.tccDriver);
        print(length, "textureAlignment", devProp.textureAlignment);
        print(length, "texturePitchAlignment", devProp.texturePitchAlignment);
        print(length, "totalConstMem", devProp.totalConstMem);
        print(length, "totalGlobalMem", devProp.totalGlobalMem);
        print(length, "unifiedAddressing", devProp.unifiedAddressing);

        //cudaUUID_t  uuid
    }
    std::cout << "---------------------------------------------------------------" << std::endl;
}
 
int main()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Device Query..." << std::endl;
    std::cout << "There are " << devCount << " CUDA devices.";
 
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
	    std::cout << std::endl << "CUDA Device number: " << i << std::endl;
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp, 3);
    }
 
    //print(length, "\nPress any key to exit...");
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


