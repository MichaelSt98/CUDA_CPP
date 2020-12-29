# 0_Simple

## [asyncAPI](asyncAPI/)

### Description

Sample: asyncAPI
Minimum spec: SM 3.0

This sample uses CUDA streams and events to overlap execution on CPU and GPU.

Key concepts:
Asynchronous Data Transfers
CUDA Streams and Events


## [cdpSimplePrint](cdpSimplePrint/)

### Description

Sample: cdpSimplePrint
Minimum spec: SM 3.5

This sample demonstrates simple printf implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

Key concepts:
CUDA Dynamic Parallelism


## [cdpSimpleQuicksort](cdpSimpleQuicksort/)

### Description

Sample: cdpSimpleQuicksort
Minimum spec: SM 3.5

This sample demonstrates simple quicksort implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

Key concepts:
CUDA Dynamic Parallelism


## [clock](clock/)

### Description

Sample: clock
Minimum spec: SM 3.0

This example shows how to use the clock function to measure the performance of block of threads of a kernel accurately.

Key concepts:
Performance Strategies


## [clock_nvrtc](clock_nvrtc/)

### Description

Sample: clock_nvrtc
Minimum spec: SM 3.0

This example shows how to use the clock function using libNVRTC to measure the performance of block of threads of a kernel accurately.

Key concepts:
Performance Strategies
Runtime Compilation


## [cppIntegration](cppIntegration/)

### Description

Sample: cppIntegration
Minimum spec: SM 3.0

This example demonstrates how to integrate CUDA into an existing C++ application, i.e. the CUDA entry point on host side is only a function which is called from C++ code and only the file containing this function is compiled with nvcc. It also demonstrates that vector types can be used from cpp.



## [cppOverload](cppOverload/)

### Description

Sample: cppOverload
Minimum spec: SM 3.0

This sample demonstrates how to use C++ function overloading on the GPU.

Key concepts:
C++ Function Overloading
CUDA Streams and Events


## [cudaOpenMP](cudaOpenMP/)

### Description

Sample: cudaOpenMP
Minimum spec: SM 3.0

This sample demonstrates how to use OpenMP API to write an application for multiple GPUs.

Key concepts:
CUDA Systems Integration
OpenMP
Multithreading


## [cudaTensorCoreGemm](cudaTensorCoreGemm/)

### Description

Sample: cudaTensorCoreGemm
Minimum spec: SM 7.0

CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in CUDA 9.

This sample demonstrates the use of the new CUDA WMMA API employing the Tensor Cores introcuced in the Volta chip family for faster matrix operations.

In addition to that, it demonstrates the use of the new CUDA function attribute cudaFuncAttributeMaxDynamicSharedMemorySize that allows the application to reserve an extended amount of shared memory than it is available by default.

Key concepts:
Matrix Multiply
WMMA
Tensor Cores


## [fp16ScalarProduct](fp16ScalarProduct/)

### Description

Sample: fp16ScalarProduct
Minimum spec: SM 5.3

Calculates scalar product of two vectors of FP16 numbers.

Key concepts:
CUDA Runtime API


## [inlinePTX](inlinePTX/)

### Description

Sample: inlinePTX
Minimum spec: SM 3.0

A simple test application that demonstrates a new CUDA 4.0 ability to embed PTX in a CUDA kernel.

Key concepts:
Performance Strategies
PTX Assembly
CUDA Driver API


## [inlinePTX_nvrtc](inlinePTX_nvrtc/)

### Description

Sample: inlinePTX_nvrtc
Minimum spec: SM 3.0

A simple test application that demonstrates a new CUDA 4.0 ability to embed PTX in a CUDA kernel.

Key concepts:
Performance Strategies
PTX Assembly
CUDA Driver API
Runtime Compilation


## [matrixMul](matrixMul/)

### Description

Sample: matrixMul
Minimum spec: SM 3.0

This sample implements matrix multiplication which makes use of shared memory to ensure data reuse, the matrix multiplication is done using tiling approach. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication.

Key concepts:
CUDA Runtime API
Linear Algebra


## [matrixMulCUBLAS](matrixMulCUBLAS/)

### Description

Sample: matrixMulCUBLAS
Minimum spec: SM 3.0

This sample implements matrix multiplication from Chapter 3 of the programming guide. To illustrate GPU performance for matrix multiply, this sample also shows how to use the new CUDA 4.0 interface for CUBLAS to demonstrate high-performance performance for matrix multiplication.

Key concepts:
CUDA Runtime API
Performance Strategies
Linear Algebra
CUBLAS


## [matrixMulDrv](matrixMulDrv/)

### Description

Sample: matrixMulDrv
Minimum spec: SM 3.0

This sample implements matrix multiplication and uses the new CUDA 4.0 kernel launch Driver API. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication. CUBLAS provides high-performance matrix multiplication.

Key concepts:
CUDA Driver API
Matrix Multiply


## [matrixMul_nvrtc](matrixMul_nvrtc/)

### Description

Sample: matrixMul_nvrtc
Minimum spec: SM 3.0

This sample implements matrix multiplication and is exactly the same as Chapter 6 of the programming guide. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication.  To illustrate GPU performance for matrix multiply, this sample also shows how to use the new CUDA 4.0 interface for CUBLAS to demonstrate high-performance performance for matrix multiplication.

Key concepts:
CUDA Runtime API
Linear Algebra
Runtime Compilation


## [simpleAssert](simpleAssert/)

### Description

Sample: simpleAssert
Minimum spec: SM 3.0

This CUDA Runtime API sample is a very basic sample that implements how to use the assert function in the device code. Requires Compute Capability 2.0 .

Key concepts:
Assert


## [simpleAssert_nvrtc](simpleAssert_nvrtc/)

### Description

Sample: simpleAssert_nvrtc
Minimum spec: SM 3.0

This CUDA Runtime API sample is a very basic sample that implements how to use the assert function in the device code. Requires Compute Capability 2.0 .

Key concepts:
Assert
Runtime Compilation


## [simpleAtomicIntrinsics](simpleAtomicIntrinsics/)

### Description

Sample: simpleAtomicIntrinsics
Minimum spec: SM 3.0

A simple demonstration of global memory atomic instructions. Requires Compute Capability 2.0 or higher.

Key concepts:
Atomic Intrinsics


## [simpleAtomicIntrinsics_nvrtc](simpleAtomicIntrinsics_nvrtc/)

### Description

Sample: simpleAtomicIntrinsics_nvrtc
Minimum spec: SM 3.0

A simple demonstration of global memory atomic instructions.This sample makes use of NVRTC for Runtime Compilation.

Key concepts:
Atomic Intrinsics
Runtime Compilation


## [simpleCallback](simpleCallback/)

### Description

Sample: simpleCallback
Minimum spec: SM 3.0

This sample implements multi-threaded heterogeneous computing workloads with the new CPU callbacks for CUDA streams and events introduced with CUDA 5.0.

Key concepts:
CUDA Streams
Callback Functions
Multithreading


## [simpleCooperativeGroups](simpleCooperativeGroups/)

### Description

Sample: simpleCooperativeGroups
Minimum spec: SM 3.0

This sample is a simple code that illustrates basic usage of cooperative groups within the thread block.

Key concepts:
Cooperative Groups


## [simpleCubemapTexture](simpleCubemapTexture/)

### Description

Sample: simpleCubemapTexture
Minimum spec: SM 3.0

Simple example that demonstrates how to use a new CUDA 4.1 feature to support cubemap Textures in CUDA C.

Key concepts:
Texture
Volume Processing


## [simpleCudaGraphs](simpleCudaGraphs/)

### Description

Sample: simpleCudaGraphs
Minimum spec: SM 3.0

A demonstration of CUDA Graphs creation, instantiation and launch using Graphs APIs and Stream Capture APIs.

Key concepts:
CUDA Graphs
Stream Capture


## [simpleIPC](simpleIPC/)

### Description

Sample: simpleIPC
Minimum spec: SM 3.0

This CUDA Runtime API sample is a very basic sample that demonstrates Inter Process Communication with one process per GPU for computation.  Requires Compute Capability 2.0 or higher and a Linux Operating System

Key concepts:
CUDA Systems Integration
Peer to Peer
InterProcess Communication


## [simpleLayeredTexture](simpleLayeredTexture/)

### Description

Sample: simpleLayeredTexture
Minimum spec: SM 3.0

Simple example that demonstrates how to use a new CUDA 4.0 feature to support layered Textures in CUDA C.

Key concepts:
Texture
Volume Processing


## [simpleMPI](simpleMPI/)

### Description

Sample: simpleMPI
Minimum spec: SM 3.0

Simple example demonstrating how to use MPI in combination with CUDA.

Key concepts:
CUDA Systems Integration
MPI
Multithreading


## [simpleMultiCopy](simpleMultiCopy/)

### Description

Sample: simpleMultiCopy
Minimum spec: SM 3.0

Supported in GPUs with Compute Capability 1.1, overlapping compute with one memcopy is possible from the host system.  For Quadro and Tesla GPUs with Compute Capability 2.0, a second overlapped copy operation in either direction at full speed is possible (PCI-e is symmetric).  This sample illustrates the usage of CUDA streams to achieve overlapping of kernel execution with data copies to and from the device.

Key concepts:
CUDA Streams and Events
Asynchronous Data Transfers
Overlap Compute and Copy
GPU Performance


## [simpleMultiGPU](simpleMultiGPU/)

### Description

Sample: simpleMultiGPU
Minimum spec: SM 3.0

This application demonstrates how to use the new CUDA 4.0 API for CUDA context management and multi-threaded access to run CUDA kernels on multiple-GPUs.

Key concepts:
Asynchronous Data Transfers
CUDA Streams and Events
Multithreading
Multi-GPU


## [simpleOccupancy](simpleOccupancy/)

### Description

Sample: simpleOccupancy
Minimum spec: SM 3.0

This sample demonstrates the basic usage of the CUDA occupancy calculator and occupancy-based launch configurator APIs by launching a kernel with the launch configurator, and measures the utilization difference against a manually configured launch.

Key concepts:
Occupancy Calculator


## [simpleP2P](simpleP2P/)

### Description

Sample: simpleP2P
Minimum spec: SM 3.0

This application demonstrates CUDA APIs that support Peer-To-Peer (P2P) copies, Peer-To-Peer (P2P) addressing, and Unified Virtual Memory Addressing (UVA) between multiple GPUs. In general, P2P is supported between two same GPUs with some exceptions, such as some Tesla and Quadro GPUs.

Key concepts:
Performance Strategies
Asynchronous Data Transfers
Unified Virtual Address Space
Peer to Peer Data Transfers
Multi-GPU


## [simplePitchLinearTexture](simplePitchLinearTexture/)

### Description

Sample: simplePitchLinearTexture
Minimum spec: SM 3.0

Use of Pitch Linear Textures

Key concepts:
Texture
Image Processing


## [simplePrintf](simplePrintf/)

### Description

Sample: simplePrintf
Minimum spec: SM 3.0

This CUDA Runtime API sample is a very basic sample that implements how to use the printf function in the device code. Specifically, for devices with compute capability less than 2.0, the function cuPrintf is called; otherwise, printf can be used directly.

Key concepts:
Debugging


## [simpleSeparateCompilation](simpleSeparateCompilation/)

### Description

Sample: simpleSeparateCompilation
Minimum spec: SM 3.0

This sample demonstrates a CUDA 5.0 feature, the ability to create a GPU device static library and use it within another CUDA kernel.  This example demonstrates how to pass in a GPU device function (from the GPU device static library) as a function pointer to be called.  This sample requires devices with compute capability 2.0 or higher.

Key concepts:
Separate Compilation


## [simpleStreams](simpleStreams/)

### Description

Sample: simpleStreams
Minimum spec: SM 3.0

This sample uses CUDA streams to overlap kernel executions with memory copies between the host and a GPU device.  This sample uses a new CUDA 4.0 feature that supports pinning of generic host memory.  Requires Compute Capability 2.0 or higher.

Key concepts:
Asynchronous Data Transfers
CUDA Streams and Events


## [simpleSurfaceWrite](simpleSurfaceWrite/)

### Description

Sample: simpleSurfaceWrite
Minimum spec: SM 3.0

Simple example that demonstrates the use of 2D surface references (Write-to-Texture)

Key concepts:
Texture
Surface Writes
Image Processing


## [simpleTemplates](simpleTemplates/)

### Description

Sample: simpleTemplates
Minimum spec: SM 3.0

This sample is a templatized version of the template project. It also shows how to correctly templatize dynamically allocated shared memory arrays.

Key concepts:
C++ Templates


## [simpleTemplates_nvrtc](simpleTemplates_nvrtc/)

### Description

Sample: simpleTemplates_nvrtc
Minimum spec: SM 3.0

This sample is a templatized version of the template project. It also shows how to correctly templatize dynamically allocated shared memory arrays.

Key concepts:
C++ Templates
Runtime Compilation


## [simpleTexture](simpleTexture/)

### Description

Sample: simpleTexture
Minimum spec: SM 3.0

Simple example that demonstrates use of Textures in CUDA.

Key concepts:
CUDA Runtime API
Texture
Image Processing


## [simpleTextureDrv](simpleTextureDrv/)

### Description

Sample: simpleTextureDrv
Minimum spec: SM 3.0

Simple example that demonstrates use of Textures in CUDA.  This sample uses the new CUDA 4.0 kernel launch Driver API.

Key concepts:
CUDA Driver API
Texture
Image Processing


## [simpleVoteIntrinsics](simpleVoteIntrinsics/)

### Description

Sample: simpleVoteIntrinsics
Minimum spec: SM 3.0

Simple program which demonstrates how to use the Vote (any, all) intrinsic instruction in a CUDA kernel.  Requires Compute Capability 2.0 or higher.

Key concepts:
Vote Intrinsics


## [simpleVoteIntrinsics_nvrtc](simpleVoteIntrinsics_nvrtc/)

### Description

Sample: simpleVoteIntrinsics_nvrtc
Minimum spec: SM 3.0

Simple program which demonstrates how to use the Vote (any, all) intrinsic instruction in a CUDA kernel with runtime compilation using NVRTC APIs. Requires Compute Capability 2.0 or higher.

Key concepts:
Vote Intrinsics
CUDA Driver API
Runtime Compilation


## [simpleZeroCopy](simpleZeroCopy/)

### Description

Sample: simpleZeroCopy
Minimum spec: SM 3.0

This sample illustrates how to use Zero MemCopy, kernels can read and write directly to pinned system memory.

Key concepts:
Performance Strategies
Pinned System Paged Memory
Vector Addition


## [systemWideAtomics](systemWideAtomics/)

### Description

Sample: systemWideAtomics
Minimum spec: SM 6.0

A simple demonstration of system wide atomic instructions.

Key concepts:
Atomic Intrinsics
Unified Memory


## [template](template/)

### Description

Sample: template
Minimum spec: SM 3.0

A trivial template project that can be used as a starting point to create new CUDA projects.

Key concepts:
Device Memory Allocation


## [UnifiedMemoryStreams](UnifiedMemoryStreams/)

### Description

Sample: UnifiedMemoryStreams
Minimum spec: SM 3.0

This sample demonstrates the use of OpenMP and streams with Unified Memory on a single GPU.

Key concepts:
CUDA Systems Integration
OpenMP
CUBLAS
Multithreading
Unified Memory
CUDA Streams and Events


## [vectorAdd](vectorAdd/)

### Description

Sample: vectorAdd
Minimum spec: SM 3.0

This CUDA Runtime API sample is a very basic sample that implements element by element vector addition. It is the same as the sample illustrating Chapter 3 of the programming guide with some additions like error checking.

Key concepts:
CUDA Runtime API
Vector Addition


## [vectorAddDrv](vectorAddDrv/)

### Description

Sample: vectorAddDrv
Minimum spec: SM 3.0

This Vector Addition sample is a basic sample that is implemented element by element.  It is the same as the sample illustrating Chapter 3 of the programming guide with some additions like error checking.   This sample also uses the new CUDA 4.0 kernel launch Driver API.

Key concepts:
CUDA Driver API
Vector Addition


## [vectorAdd_nvrtc](vectorAdd_nvrtc/)

### Description

Sample: vectorAdd_nvrtc
Minimum spec: SM 3.0

This CUDA Driver API sample uses NVRTC for runtime compilation of vector addition kernel. Vector addition kernel demonstrated is the same as the sample illustrating Chapter 3 of the programming guide.

Key concepts:
CUDA Driver API
Vector Addition
Runtime Compilation


