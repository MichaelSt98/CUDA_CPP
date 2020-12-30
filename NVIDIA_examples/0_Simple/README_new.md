## [asyncAPI](asyncAPI/)

### Description

Sample: asyncAPI
Minimum spec: SM 3.0

This sample uses CUDA streams and events to overlap execution on CPU and GPU.

Key concepts:
Asynchronous Data Transfers
CUDA Streams and Events

### Sample output

[./asyncAPI/asyncAPI] - Starting...
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

CUDA device [GeForce GTX 1080 Ti]
time spent executing by the GPU: 20.34
time spent by CPU in CUDA calls: 0.03
CPU executed 80223 iterations while waiting for GPU to finish


## [cdpSimplePrint](cdpSimplePrint/)

### Description

Sample: cdpSimplePrint
Minimum spec: SM 3.5

This sample demonstrates simple printf implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

Key concepts:
CUDA Dynamic Parallelism

### Sample output

starting Simple Print (CUDA Dynamic Parallelism)
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

***************************************************************************
The CPU launches 2 blocks of 2 threads each. On the device each thread will
launch 2 blocks of 2 threads each. The GPU we will do that recursively
until it reaches max_depth=2

In total 2+8=10 blocks are launched!!! (8 from the GPU)
***************************************************************************

Launching cdp_kernel() with CUDA Dynamic Parallelism:

BLOCK 0 launched by the host
BLOCK 1 launched by the host
|  BLOCK 2 launched by thread 0 of block 0
|  BLOCK 5 launched by thread 0 of block 1
|  BLOCK 4 launched by thread 0 of block 1
|  BLOCK 3 launched by thread 0 of block 0
|  BLOCK 6 launched by thread 1 of block 1
|  BLOCK 7 launched by thread 1 of block 1
|  BLOCK 9 launched by thread 1 of block 0
|  BLOCK 8 launched by thread 1 of block 0


## [cdpSimpleQuicksort](cdpSimpleQuicksort/)

### Description

Sample: cdpSimpleQuicksort
Minimum spec: SM 3.5

This sample demonstrates simple quicksort implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

Key concepts:
CUDA Dynamic Parallelism

### Sample output

GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

Initializing data:
Running quicksort on 128 elements
Launching kernel on the GPU
Validating results: OK


## [clock](clock/)

### Description

Sample: clock
Minimum spec: SM 3.0

This example shows how to use the clock function to measure the performance of block of threads of a kernel accurately.

Key concepts:
Performance Strategies

### Sample output

CUDA Clock sample
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

Average clocks/block = 4646.218750


## [clock_nvrtc](clock_nvrtc/)

### Description

Sample: clock_nvrtc
Minimum spec: SM 3.0

This example shows how to use the clock function using libNVRTC to measure the performance of block of threads of a kernel accurately.

Key concepts:
Performance Strategies
Runtime Compilation

### Sample output

CUDA Clock sample

error: unable to open Problems with clock_nvrtc


## [cppIntegration](cppIntegration/)

### Description

Sample: cppIntegration
Minimum spec: SM 3.0

This example demonstrates how to integrate CUDA into an existing C++ application, i.e. the CUDA entry point on host side is only a function which is called from C++ code and only the file containing this function is compiled with nvcc. It also demonstrates that vector types can be used from cpp.


### Sample output

GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

Hello World.
Hello World.


## [cppOverload](cppOverload/)

### Description

Sample: cppOverload
Minimum spec: SM 3.0

This sample demonstrates how to use C++ function overloading on the GPU.

Key concepts:
C++ Function Overloading
CUDA Streams and Events

### Sample output

C++ Function Overloading starting...
DevicecheckCudaErrors Count: 4
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

Shared Size:   1024
Constant Size: 0
Local Size:    0
Max Threads Per Block: 1024
Number of Registers: 10
PTX Version: 61
Binary Version: 61
simple_kernel(const int *pIn, int *pOut, int a) PASSED

Shared Size:   2048
Constant Size: 0
Local Size:    0
Max Threads Per Block: 1024
Number of Registers: 12
PTX Version: 61
Binary Version: 61
simple_kernel(const int2 *pIn, int *pOut, int a) PASSED

Shared Size:   2048
Constant Size: 0
Local Size:    0
Max Threads Per Block: 1024
Number of Registers: 11
PTX Version: 61
Binary Version: 61
simple_kernel(const int *pIn1, const int *pIn2, int *pOut, int a) PASSED



## [cudaOpenMP](cudaOpenMP/)

### Description

Sample: cudaOpenMP
Minimum spec: SM 3.0

This sample demonstrates how to use OpenMP API to write an application for multiple GPUs.

Key concepts:
CUDA Systems Integration
OpenMP
Multithreading

### Sample output

./cudaOpenMP/cudaOpenMP Starting...

number of host CPUs:	16
number of CUDA devices:	4
   0: GeForce GTX 1080 Ti
   1: GeForce GTX 1080 Ti
   2: GeForce GTX 1080 Ti
   3: GeForce GTX 1080 Ti
---------------------------
CPU thread 0 (of 4) uses CUDA device 0
CPU thread 2 (of 4) uses CUDA device 2
CPU thread 3 (of 4) uses CUDA device 3
CPU thread 1 (of 4) uses CUDA device 1
---------------------------


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

### Sample output

Initializing...
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

cudaTensorCoreGemm requires requires SM 7.0 or higher to use Tensor Cores.  Exiting...
Problems with cudaTensorCoreGemm


## [fp16ScalarProduct](fp16ScalarProduct/)

### Description

Sample: fp16ScalarProduct
Minimum spec: SM 5.3

Calculates scalar product of two vectors of FP16 numbers.

Key concepts:
CUDA Runtime API

### Sample output

GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

Result native operators	: 656990.000000 
Result intrinsics	: 656990.000000 
&&&& fp16ScalarProduct PASSED


## [inlinePTX](inlinePTX/)

### Description

Sample: inlinePTX
Minimum spec: SM 3.0

A simple test application that demonstrates a new CUDA 4.0 ability to embed PTX in a CUDA kernel.

Key concepts:
Performance Strategies
PTX Assembly
CUDA Driver API

### Sample output

CUDA inline PTX assembler sample
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

Test Successful.


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

### Sample output

CUDA inline PTX assembler sample

error: unable to open Problems with inlinePTX_nvrtc


## [matrixMul](matrixMul/)

### Description

Sample: matrixMul
Minimum spec: SM 3.0

This sample implements matrix multiplication which makes use of shared memory to ensure data reuse, the matrix multiplication is done using tiling approach. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication.

Key concepts:
CUDA Runtime API
Linear Algebra

### Sample output

[Matrix Multiply Using CUDA] - Starting...
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

MatrixA(320,320), MatrixB(640,320)
Computing result using CUDA Kernel...
done
Performance= 778.46 GFlop/s, Time= 0.168 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
Checking computed result for correctness: Result = PASS

NOTE: The CUDA Samples are not meant for performancemeasurements. Results may vary when GPU Boost is enabled.


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

### Sample output

[Matrix Multiply CUBLAS] - Starting...
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

MatrixA(640,480), MatrixB(480,320), MatrixC(640,320)
Computing result using CUBLAS...done.
Performance= 4744.65 GFlop/s, Time= 0.041 msec, Size= 196608000 Ops
Computing result using host CPU...done.
Comparing CUBLAS Matrix Multiply with CPU results: PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.


## [matrixMulDrv](matrixMulDrv/)

### Description

Sample: matrixMulDrv
Minimum spec: SM 3.0

This sample implements matrix multiplication and uses the new CUDA 4.0 kernel launch Driver API. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication. CUBLAS provides high-performance matrix multiplication.

Key concepts:
CUDA Driver API
Matrix Multiply

### Sample output

[ matrixMulDrv (Driver API) ]
> Using CUDA Device [0]: GeForce GTX 1080 Ti
> GPU Device has SM 6.1 compute capability
  Total amount of global memory:     11721506816 bytes
  64-bit Memory Address:             YES
> findModulePath <../0_Simple/matrixMulDrv/data/matrixMul_kernel64.ptx>
> initCUDA loading module: <../0_Simple/matrixMulDrv/data/matrixMul_kernel64.ptx>
> PTX JIT log:

Processing time: 1.766000 (ms)
Checking computed result for correctness: Result = PASS

NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.


## [matrixMul_nvrtc](matrixMul_nvrtc/)

### Description

Sample: matrixMul_nvrtc
Minimum spec: SM 3.0

This sample implements matrix multiplication and is exactly the same as Chapter 6 of the programming guide. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication.  To illustrate GPU performance for matrix multiply, this sample also shows how to use the new CUDA 4.0 interface for CUBLAS to demonstrate high-performance performance for matrix multiplication.

Key concepts:
CUDA Runtime API
Linear Algebra
Runtime Compilation

### Sample output

[Matrix Multiply Using CUDA] - Starting...
MatrixA(320,320), MatrixB(640,320)

error: unable to open Problems with matrixMul_nvrtc


## [simpleAssert](simpleAssert/)

### Description

Sample: simpleAssert
Minimum spec: SM 3.0

This CUDA Runtime API sample is a very basic sample that implements how to use the assert function in the device code. Requires Compute Capability 2.0 .

Key concepts:
Assert

### Sample output

simpleAssert.cu:47: void testKernel(int): block: [1,0,0], thread: [28,0,0] Assertion `gtid < N` failed.
simpleAssert.cu:47: void testKernel(int): block: [1,0,0], thread: [29,0,0] Assertion `gtid < N` failed.
simpleAssert.cu:47: void testKernel(int): block: [1,0,0], thread: [30,0,0] Assertion `gtid < N` failed.
simpleAssert.cu:47: void testKernel(int): block: [1,0,0], thread: [31,0,0] Assertion `gtid < N` failed.
simpleAssert starting...
OS_System_Type.release = 4.15.0-117-generic
OS Info: <#118-Ubuntu SMP Fri Sep 4 20:02:41 UTC 2020>

GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

Launch kernel to generate assertion failures

-- Begin assert output


-- End assert output

Device assert failed as expected, CUDA error message is: device-side assert triggered

simpleAssert completed, returned OK


## [simpleAssert_nvrtc](simpleAssert_nvrtc/)

### Description

Sample: simpleAssert_nvrtc
Minimum spec: SM 3.0

This CUDA Runtime API sample is a very basic sample that implements how to use the assert function in the device code. Requires Compute Capability 2.0 .

Key concepts:
Assert
Runtime Compilation

### Sample output

simpleAssert_nvrtc starting...
Launch kernel to generate assertion failures

error: unable to open Problems with simpleAssert_nvrtc


## [simpleAtomicIntrinsics](simpleAtomicIntrinsics/)

### Description

Sample: simpleAtomicIntrinsics
Minimum spec: SM 3.0

A simple demonstration of global memory atomic instructions. Requires Compute Capability 2.0 or higher.

Key concepts:
Atomic Intrinsics

### Sample output

simpleAtomicIntrinsics starting...
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

> GPU device has 28 Multi-Processors, SM 6.1 compute capabilities

Processing time: 114.567001 (ms)
simpleAtomicIntrinsics completed, returned OK


## [simpleAtomicIntrinsics_nvrtc](simpleAtomicIntrinsics_nvrtc/)

### Description

Sample: simpleAtomicIntrinsics_nvrtc
Minimum spec: SM 3.0

A simple demonstration of global memory atomic instructions.This sample makes use of NVRTC for Runtime Compilation.

Key concepts:
Atomic Intrinsics
Runtime Compilation

### Sample output

simpleAtomicIntrinsics_nvrtc starting...

error: unable to open Problems with simpleAtomicIntrinsics_nvrtc


## [simpleCallback](simpleCallback/)

### Description

Sample: simpleCallback
Minimum spec: SM 3.0

This sample implements multi-threaded heterogeneous computing workloads with the new CPU callbacks for CUDA streams and events introduced with CUDA 5.0.

Key concepts:
CUDA Streams
Callback Functions
Multithreading

### Sample output

Starting simpleCallback
Found 4 CUDA capable GPUs
GPU[0] GeForce GTX 1080 Ti supports SM 6.1, capable GPU Callback Functions
GPU[1] GeForce GTX 1080 Ti supports SM 6.1, capable GPU Callback Functions
GPU[2] GeForce GTX 1080 Ti supports SM 6.1, capable GPU Callback Functions
GPU[3] GeForce GTX 1080 Ti supports SM 6.1, capable GPU Callback Functions
4 GPUs available to run Callback Functions
Starting 8 heterogeneous computing workloads
Total of 8 workloads finished:
Success


## [simpleCooperativeGroups](simpleCooperativeGroups/)

### Description

Sample: simpleCooperativeGroups
Minimum spec: SM 3.0

This sample is a simple code that illustrates basic usage of cooperative groups within the thread block.

Key concepts:
Cooperative Groups

### Sample output


Launching a single block with 64 threads...

 Sum of all ranks 0..63 in threadBlockGroup is 2016 (expected 2016)

 Now creating 4 groups, each of size 16 threads:

   Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)
   Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)
   Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)
   Sum of all ranks 0..15 in this tiledPartition16 group is 120 (expected 120)

...Done.



## [simpleCubemapTexture](simpleCubemapTexture/)

### Description

Sample: simpleCubemapTexture
Minimum spec: SM 3.0

Simple example that demonstrates how to use a new CUDA 4.1 feature to support cubemap Textures in CUDA C.

Key concepts:
Texture
Volume Processing

### Sample output

GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

CUDA device [GeForce GTX 1080 Ti] has 28 Multi-Processors SM 6.1
Covering Cubemap data array of 64~3 x 1: Grid size is 8 x 8, each block has 8 x 8 threads
Processing time: 2.169 msec
11.33 Mtexlookups/sec
Comparing kernel output to expected data


## [simpleCudaGraphs](simpleCudaGraphs/)

### Description

Sample: simpleCudaGraphs
Minimum spec: SM 3.0

A demonstration of CUDA Graphs creation, instantiation and launch using Graphs APIs and Stream Capture APIs.

Key concepts:
CUDA Graphs
Stream Capture

### Sample output

GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

16777216 elements
threads per block  = 512
Graph Launch iterations = 3

Num of nodes in the graph created manually = 7
[cudaGraphsManual] Host callback final reduced sum = 0.996214
[cudaGraphsManual] Host callback final reduced sum = 0.996214
[cudaGraphsManual] Host callback final reduced sum = 0.996214
Cloned Graph Output.. 
[cudaGraphsManual] Host callback final reduced sum = 0.996214
[cudaGraphsManual] Host callback final reduced sum = 0.996214
[cudaGraphsManual] Host callback final reduced sum = 0.996214

Num of nodes in the graph created using stream capture API = 7
[cudaGraphsUsingStreamCapture] Host callback final reduced sum = 0.996214
[cudaGraphsUsingStreamCapture] Host callback final reduced sum = 0.996214
[cudaGraphsUsingStreamCapture] Host callback final reduced sum = 0.996214
Cloned Graph Output.. 
[cudaGraphsUsingStreamCapture] Host callback final reduced sum = 0.996214
[cudaGraphsUsingStreamCapture] Host callback final reduced sum = 0.996214
[cudaGraphsUsingStreamCapture] Host callback final reduced sum = 0.996214


## [simpleIPC](simpleIPC/)

### Description

Sample: simpleIPC
Minimum spec: SM 3.0

This CUDA Runtime API sample is a very basic sample that demonstrates Inter Process Communication with one process per GPU for computation.  Requires Compute Capability 2.0 or higher and a Linux Operating System

Key concepts:
CUDA Systems Integration
Peer to Peer
InterProcess Communication

### Sample output


Checking for multiple GPUs...
CUDA-capable device count: 4

Searching for UVA capable devices...
> GPU0 = "GeForce GTX 1080 Ti" IS capable of UVA
> GPU1 = "GeForce GTX 1080 Ti" IS capable of UVA
> GPU2 = "GeForce GTX 1080 Ti" IS capable of UVA
> GPU3 = "GeForce GTX 1080 Ti" IS capable of UVA

Checking GPU(s) for support of peer to peer memory access...
> Two-way peer access between GPU0 and GPU1: YES
> Two-way peer access between GPU0 and GPU2: YES
> Two-way peer access between GPU0 and GPU3: YES
Data check error at index 0 in process 1!: 1804289383,    0

Spawning processes and assigning GPUs...
> Process   0 -> GPU0

Launching kernels...
Checking test results...
Problems with simpleIPC


## [simpleLayeredTexture](simpleLayeredTexture/)

### Description

Sample: simpleLayeredTexture
Minimum spec: SM 3.0

Simple example that demonstrates how to use a new CUDA 4.0 feature to support layered Textures in CUDA C.

Key concepts:
Texture
Volume Processing

### Sample output


Spawning processes and assigning GPUs...
> Process   2 -> GPU2
> Process   2: Run kernel on GPU2, taking source data from and writing results to process 0, GPU0...

Spawning processes and assigning GPUs...
> Process   3 -> GPU3
> Process   3: Run kernel on GPU3, taking source data from and writing results to process 0, GPU0...

Spawning processes and assigning GPUs...
> Process   1 -> GPU1
> Process   1: Run kernel on GPU1, taking source data from and writing results to process 0, GPU0...
[simpleLayeredTexture] - Starting...
GPU Device 0: "GeForce GTX 1080 Ti" with compute capability 6.1

CUDA device [GeForce GTX 1080 Ti] has 28 Multi-Processors SM 6.1
Covering 2D data array of 512 x 512: Grid size is 64 x 64, each block has 8 x 8 threads
Processing time: 2.232 msec
587.24 Mtexlookups/sec
Comparing kernel output to expected data


## [simpleMPI](simpleMPI/)

### Description

Sample: simpleMPI
Minimum spec: SM 3.0

Simple example demonstrating how to use MPI in combination with CUDA.

Key concepts:
CUDA Systems Integration
MPI
Multithreading

### Sample output

Running on 1 nodes
Average of square roots is: 0.667242
PASSED


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

### Sample output

[simpleMultiCopy] - Starting...
> Using CUDA device [0]: GeForce GTX 1080 Ti
[GeForce GTX 1080 Ti] has 28 MP(s) x 128 (Cores/MP) = 3584 (Cores)
> Device name: GeForce GTX 1080 Ti
> CUDA Capability 6.1 hardware with 28 multi-processors
> scale_factor = 1.00
> array_size   = 4194304


Relevant properties of this CUDA device
(X) Can overlap one CPU<>GPU data transfer with GPU kernel execution (device property "deviceOverlap")
(X) Can overlap two CPU<>GPU data transfers with GPU kernel execution
    (Compute Capability >= 2.0 AND (Tesla product OR Quadro 4000/5000/6000/K5000)

Measured timings (throughput):
 Memcpy host to device	: 2.547488 ms (6.585788 GB/s)
 Memcpy device to host	: 2.485056 ms (6.751243 GB/s)
 Kernel			: 0.194976 ms (860.475947 GB/s)

Theoretical limits for speedup gained from overlapped data transfers:
No overlap at all (transfer-kernel-transfer): 5.227520 ms 
Compute can overlap with one transfer: 5.032544 ms
Compute can overlap with both data transfers: 2.547488 ms

Average measured timings over 10 repetitions:
 Avg. time when execution fully serialized	: 5.235814 ms
 Avg. time when overlapped using 4 streams	: 3.094118 ms
 Avg. speedup gained (serialized - overlapped)	: 2.141696 ms

Measured throughput:
 Fully serialized execution		: 6.408637 GB/s
 Overlapped using 4 streams		: 10.844586 GB/s


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

### Sample output

Starting simpleMultiGPU
CUDA-capable device count: 4
Generating input data...

Computing with 4 GPUs...
  GPU Processing time: 17.414000 (ms)

Computing with Host CPU...

Comparing GPU and Host CPU results...
  GPU sum: 16777304.000000
  CPU sum: 16777294.395033
  Relative difference: 5.724980E-07 



## [simpleOccupancy](simpleOccupancy/)

### Description

Sample: simpleOccupancy
Minimum spec: SM 3.0

This sample demonstrates the basic usage of the CUDA occupancy calculator and occupancy-based launch configurator APIs by launching a kernel with the launch configurator, and measures the utilization difference against a manually configured launch.

Key concepts:
Occupancy Calculator

### Sample output

starting Simple Occupancy

[ Manual configuration with 32 threads per block ]
Potential occupancy: 50%
Elapsed time: 0.122016ms

[ Automatic, occupancy-based configuration ]
Suggested block size: 1024
Minimum grid size for maximum occupancy: 56
Potential occupancy: 100%
Elapsed time: 0.096064ms

Test PASSED



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

### Sample output

