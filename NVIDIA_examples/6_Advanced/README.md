# 6_Advanced

## [alignedTypes](alignedTypes/)

### Description

Sample: alignedTypes
Minimum spec: SM 3.0

A simple test, showing huge access speed gap between aligned and misaligned structures.

Key concepts:
Performance Strategies


## [c++11_cuda](c++11_cuda/)

### Description

Sample: c++11_cuda
Minimum spec: SM 3.0

This sample demonstrates C++11 feature support in CUDA. It scans a input text file and prints no. of occurrences of x, y, z, w characters. 

Key concepts:
CPP11 CUDA


## [cdpAdvancedQuicksort](cdpAdvancedQuicksort/)

### Description

Sample: cdpAdvancedQuicksort
Minimum spec: SM 3.5

This sample demonstrates an advanced quicksort implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

Key concepts:
Cooperative Groups
CUDA Dynamic Parallelism


## [cdpBezierTessellation](cdpBezierTessellation/)

### Description

Sample: cdpBezierTessellation
Minimum spec: SM 3.5

This sample demonstrates bezier tessellation of lines implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

Key concepts:
CUDA Dynamic Parallelism


## [cdpQuadtree](cdpQuadtree/)

### Description

Sample: cdpQuadtree
Minimum spec: SM 3.5

This sample demonstrates Quad Trees implemented using CUDA Dynamic Parallelism.  This sample requires devices with compute capability 3.5 or higher.

Key concepts:
Cooperative Groups
CUDA Dynamic Parallelism


## [concurrentKernels](concurrentKernels/)

### Description

Sample: concurrentKernels
Minimum spec: SM 3.0

This sample demonstrates the use of CUDA streams for concurrent execution of several kernels on devices of compute capability 2.0 or higher.  Devices of compute capability 1.x will run the kernels sequentially.  It also illustrates how to introduce dependencies between CUDA streams with the new cudaStreamWaitEvent function introduced in CUDA 3.2

Key concepts:
Performance Strategies


## [conjugateGradientMultiBlockCG](conjugateGradientMultiBlockCG/)

### Description

Sample: conjugateGradientMultiBlockCG
Minimum spec: SM 6.0

This sample implements a conjugate gradient solver on GPU using Multi Block Cooperative Groups, also uses Unified Memory.

Key concepts:
Unified Memory
Linear Algebra
Cooperative Groups
MultiBlock Cooperative Groups


## [conjugateGradientMultiDeviceCG](conjugateGradientMultiDeviceCG/)

### Description

Sample: conjugateGradientMultiDeviceCG
Minimum spec: SM 6.0

This sample implements a conjugate gradient solver on multiple GPUs using Multi Device Cooperative Groups, also uses Unified Memory optimized using prefetching and usage hints.

Key concepts:
Unified Memory
Linear Algebra
Cooperative Groups
MultiDevice Cooperative Groups


## [eigenvalues](eigenvalues/)

### Description

Sample: eigenvalues
Minimum spec: SM 3.0

The computation of all or a subset of all eigenvalues is an important problem in Linear Algebra, statistics, physics, and many other fields. This sample demonstrates a parallel implementation of a bisection algorithm for the computation of all eigenvalues of a tridiagonal symmetric matrix of arbitrary size with CUDA.

Key concepts:
Linear Algebra


## [fastWalshTransform](fastWalshTransform/)

### Description

Sample: fastWalshTransform
Minimum spec: SM 3.0

Naturally(Hadamard)-ordered Fast Walsh Transform for batching vectors of arbitrary eligible lengths that are power of two in size.

Key concepts:
Linear Algebra
Data-Parallel Algorithms
Video Compression


## [FDTD3d](FDTD3d/)

### Description

Sample: FDTD3d
Minimum spec: SM 3.0

This sample applies a finite differences time domain progression stencil on a 3D surface.

Key concepts:
Performance Strategies


## [FunctionPointers](FunctionPointers/)

### Description

Sample: FunctionPointers
Minimum spec: SM 3.0

This sample illustrates how to use function pointers and implements the Sobel Edge Detection filter for 8-bit monochrome images.

Key concepts:
Graphics Interop
Image Processing


## [interval](interval/)

### Description

Sample: interval
Minimum spec: SM 3.0

Interval arithmetic operators example.  Uses various C++ features (templates and recursion).  The recursive mode requires Compute SM 2.0 capabilities.

Key concepts:
Recursion
Templates


## [lineOfSight](lineOfSight/)

### Description

Sample: lineOfSight
Minimum spec: SM 3.0

This sample is an implementation of a simple line-of-sight algorithm: Given a height map and a ray originating at some observation point, it computes all the points along the ray that are visible from the observation point. The implementation is based on the Thrust library (http://code.google.com/p/thrust/).



## [matrixMulDynlinkJIT](matrixMulDynlinkJIT/)

### Description

Sample: matrixMulDynlinkJIT
Minimum spec: SM 3.0

This sample revisits matrix multiplication using the CUDA driver API. It demonstrates how to link to CUDA driver at runtime and how to use JIT (just-in-time) compilation from PTX code. It has been written for clarity of exposition to illustrate various CUDA programming principles, not with the goal of providing the most performant generic kernel for matrix multiplication. CUBLAS provides high-performance matrix multiplication.

Key concepts:
CUDA Driver API
CUDA Dynamically Linked Library


## [mergeSort](mergeSort/)

### Description

Sample: mergeSort
Minimum spec: SM 3.0

This sample implements a merge sort (also known as Batcher's sort), algorithms belonging to the class of sorting networks. While generally subefficient on large sequences compared to algorithms with better asymptotic algorithmic complexity (i.e. merge sort or radix sort), may be the algorithms of choice for sorting batches of short- to mid-sized (key, value) array pairs. Refer to the excellent tutorial by H. W. Lang http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm

Key concepts:
Data-Parallel Algorithms


## [newdelete](newdelete/)

### Description

Sample: newdelete
Minimum spec: SM 3.0

This sample demonstrates dynamic global memory allocation through device C++ new and delete operators and virtual function declarations available with CUDA 4.0.



## [ptxjit](ptxjit/)

### Description

Sample: ptxjit
Minimum spec: SM 3.0

This sample uses the Driver API to just-in-time compile (JIT) a Kernel from PTX code. Additionally, this sample demonstrates the seamless interoperability capability of the CUDA Runtime and CUDA Driver API calls.  For CUDA 5.5, this sample shows how to use cuLink* functions to link PTX assembly using the CUDA driver at runtime.

Key concepts:
CUDA Driver API


## [radixSortThrust](radixSortThrust/)

### Description

Sample: radixSortThrust
Minimum spec: SM 3.0

This sample demonstrates a very fast and efficient parallel radix sort uses Thrust library. The included RadixSort class can sort either key-value pairs (with float or unsigned integer keys) or keys only.

Key concepts:
Data-Parallel Algorithms
Performance Strategies


## [reduction](reduction/)

### Description

Sample: reduction
Minimum spec: SM 3.0

A parallel sum reduction that computes the sum of a large arrays of values.  This sample demonstrates several important optimization strategies for 1:Data-Parallel Algorithms like reduction.

Key concepts:
Data-Parallel Algorithms
Performance Strategies


## [reductionMultiBlockCG](reductionMultiBlockCG/)

### Description

Sample: reductionMultiBlockCG
Minimum spec: SM 6.0

This sample demonstrates single pass reduction using Multi Block Cooperative Groups.  This sample requires devices with compute capability 6.0 or higher having compute preemption.

Key concepts:
Cooperative Groups
MultiBlock Cooperative Groups


## [scalarProd](scalarProd/)

### Description

Sample: scalarProd
Minimum spec: SM 3.0

This sample calculates scalar products of a given set of input vector pairs.

Key concepts:
Linear Algebra


## [scan](scan/)

### Description

Sample: scan
Minimum spec: SM 3.0

This example demonstrates an efficient CUDA implementation of parallel prefix sum, also known as "scan".  Given an array of numbers, scan computes a new array in which each element is the sum of all the elements before it in the input array.

Key concepts:
Data-Parallel Algorithms
Performance Strategies


## [segmentationTreeThrust](segmentationTreeThrust/)

### Description

Sample: segmentationTreeThrust
Minimum spec: SM 3.0

This sample demonstrates an approach to the image segmentation trees construction.  This method is based on Boruvka's MST algorithm.

Key concepts:
Data-Parallel Algorithms
Performance Strategies


## [shfl_scan](shfl_scan/)

### Description

Sample: shfl_scan
Minimum spec: SM 3.0

This example demonstrates how to use the shuffle intrinsic __shfl_up to perform a scan operation across a thread block.  A GPU with Compute Capability SM 3.0. is required to run the sample

Key concepts:
Data-Parallel Algorithms
Performance Strategies


## [simpleHyperQ](simpleHyperQ/)

### Description

Sample: simpleHyperQ
Minimum spec: SM 3.0

This sample demonstrates the use of CUDA streams for concurrent execution of several kernels on devices which provide HyperQ (SM 3.5).  Devices without HyperQ (SM 2.0 and SM 3.0) will run a maximum of two kernels concurrently.

Key concepts:
CUDA Systems Integration
Performance Strategies


## [sortingNetworks](sortingNetworks/)

### Description

Sample: sortingNetworks
Minimum spec: SM 3.0

This sample implements bitonic sort and odd-even merge sort (also known as Batcher's sort), algorithms belonging to the class of sorting networks. While generally subefficient, for large sequences compared to algorithms with better asymptotic algorithmic complexity (i.e. merge sort or radix sort), this may be the preferred algorithms of choice for sorting batches of short-sized to mid-sized (key, value) array pairs. Refer to an excellent tutorial by H. W. Lang http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm

Key concepts:
Data-Parallel Algorithms


## [StreamPriorities](StreamPriorities/)

### Description

Sample: StreamPriorities
Minimum spec: SM 3.5

This sample demonstrates basic use of stream priorities.

Key concepts:
CUDA Streams and Events


## [threadFenceReduction](threadFenceReduction/)

### Description

Sample: threadFenceReduction
Minimum spec: SM 3.0

This sample shows how to perform a reduction operation on an array of values using the thread Fence intrinsic to produce a single value in a single kernel (as opposed to two or more kernel calls as shown in the "reduction" CUDA Sample).  Single-pass reduction requires global atomic instructions (Compute Capability 2.0 or later) and the _threadfence() intrinsic (CUDA 2.2 or later).

Key concepts:
Cooperative Groups
Data-Parallel Algorithms
Performance Strategies


## [threadMigration](threadMigration/)

### Description

Sample: threadMigration
Minimum spec: SM 3.0

Simple program illustrating how to the CUDA Context Management API and uses the new CUDA 4.0 parameter passing and CUDA launch API.  CUDA contexts can be created separately and attached independently to different threads.

Key concepts:
CUDA Driver API


## [transpose](transpose/)

### Description

Sample: transpose
Minimum spec: SM 3.0

This sample demonstrates Matrix Transpose.  Different performance are shown to achieve high performance.

Key concepts:
Performance Strategies
Linear Algebra


## [warpAggregatedAtomicsCG](warpAggregatedAtomicsCG/)

### Description

Sample: warpAggregatedAtomicsCG
Minimum spec: SM 3.0

This sample demonstrates how using Cooperative Groups (CG) to perform warp aggregated atomics, a useful technique to improve performance when many threads atomically add to a single counter.

Key concepts:
Cooperative Groups
Atomic Intrinsics


