# 7_CUDALibraries

## [batchCUBLAS](batchCUBLAS/)

### Description

Sample: batchCUBLAS
Minimum spec: SM 3.0

A CUDA Sample that demonstrates how using batched CUBLAS API calls to improve overall performance.

Key concepts:
Linear Algebra
CUBLAS Library


## [BiCGStab](BiCGStab/)

### Description

Sample: BiCGStab
Minimum spec: SM 3.0

A CUDA Sample that demonstrates Bi-Conjugate Gradient Stabilized (BiCGStab) iterative method for nonsymmetric and symmetric positive definite (s.p.d.) linear systems using CUSPARSE and CUBLAS.

Key concepts:
Linear Algebra
CUBLAS Library
CUSPARSE Library


## [boundSegmentsNPP](boundSegmentsNPP/)

### Description

Sample: boundSegmentsNPP
Minimum spec: SM 3.0

An NPP CUDA Sample that demonstrates using nppiLabelMarkers to generate connected region segment labels in an 8-bit grayscale image then compressing the sparse list of generated labels into the minimum number of uniquely labeled regions in the image using nppiCompressMarkerLabels.  Finally a boundary is added surrounding each segmented region in the image using nppiBoundSegments.

Key concepts:
Performance Strategies
Image Processing
NPP Library


## [boxFilterNPP](boxFilterNPP/)

### Description

Sample: boxFilterNPP
Minimum spec: SM 3.0

A NPP CUDA Sample that demonstrates how to use NPP FilterBox function to perform a Box Filter.

Key concepts:
Performance Strategies
Image Processing
NPP Library


## [cannyEdgeDetectorNPP](cannyEdgeDetectorNPP/)

### Description

Sample: cannyEdgeDetectorNPP
Minimum spec: SM 3.0

An NPP CUDA Sample that demonstrates the recommended parameters to use with the nppiFilterCannyBorder_8u_C1R Canny Edge Detection image filter function. This function expects a single channel 8-bit grayscale input image. You can generate a grayscale image from a color image by first calling nppiColorToGray() or nppiRGBToGray(). The Canny Edge Detection function combines and improves on the techniques required to produce an edge detection image using multiple steps.

Key concepts:
Performance Strategies
Image Processing
NPP Library


## [common](common/)

### Description



## [conjugateGradient](conjugateGradient/)

### Description

Sample: conjugateGradient
Minimum spec: SM 3.0

This sample implements a conjugate gradient solver on GPU using CUBLAS and CUSPARSE library.

Key concepts:
Linear Algebra
CUBLAS Library
CUSPARSE Library


## [conjugateGradientPrecond](conjugateGradientPrecond/)

### Description

Sample: conjugateGradientPrecond
Minimum spec: SM 3.0

This sample implements a preconditioned conjugate gradient solver on GPU using CUBLAS and CUSPARSE library.

Key concepts:
Linear Algebra
CUBLAS Library
CUSPARSE Library


## [conjugateGradientUM](conjugateGradientUM/)

### Description

Sample: conjugateGradientUM
Minimum spec: SM 3.0

This sample implements a conjugate gradient solver on GPU using CUBLAS and CUSPARSE library, using Unified Memory

Key concepts:
Unified Memory
Linear Algebra
CUBLAS Library
CUSPARSE Library


## [cuHook](cuHook/)

### Description

Sample: cuHook
Minimum spec: SM 3.0

This sample demonstrates how to build and use an intercept library with CUDA. The library has to be loaded via LD_PRELOAD, e.g. LD_PRELOAD=<full_path>/libcuhook.so.1 ./cuHook



## [cuSolverDn_LinearSolver](cuSolverDn_LinearSolver/)

### Description

Sample: cuSolverDn_LinearSolver
Minimum spec: SM 3.0

A CUDA Sample that demonstrates cuSolverDN's LU, QR and Cholesky factorization.

Key concepts:
Linear Algebra
CUSOLVER Library


## [cuSolverRf](cuSolverRf/)

### Description

Sample: cuSolverRf
Minimum spec: SM 3.0

A CUDA Sample that demonstrates cuSolver's refactorization library - CUSOLVERRF.

Key concepts:
Linear Algebra
CUSOLVER Library


## [cuSolverSp_LinearSolver](cuSolverSp_LinearSolver/)

### Description

Sample: cuSolverSp_LinearSolver
Minimum spec: SM 3.0

A CUDA Sample that demonstrates cuSolverSP's LU, QR and Cholesky factorization.

Key concepts:
Linear Algebra
CUSOLVER Library


## [cuSolverSp_LowlevelCholesky](cuSolverSp_LowlevelCholesky/)

### Description

Sample: cuSolverSp_LowlevelCholesky
Minimum spec: SM 3.0

A CUDA Sample that demonstrates Cholesky factorization using cuSolverSP's low level APIs.

Key concepts:
Linear Algebra
CUSOLVER Library


## [cuSolverSp_LowlevelQR](cuSolverSp_LowlevelQR/)

### Description

Sample: cuSolverSp_LowlevelQR
Minimum spec: SM 3.0

A CUDA Sample that demonstrates QR factorization using cuSolverSP's low level APIs.

Key concepts:
Linear Algebra
CUSOLVER Library


## [FilterBorderControlNPP](FilterBorderControlNPP/)

### Description

Sample: FilterBorderControlNPP
Minimum spec: SM 3.0

This NPP CUDA Sample demonstrates how any border version of an NPP filtering function can be used in the most common mode (with border control enabled), can be used to duplicate the results of the equivalent non-border version of the NPP function, and can be used to enable and disable border control on various source image edges depending on what portion of the source image is being used as input.

Key concepts:
Performance Strategies
Image Processing
NPP Library


## [freeImageInteropNPP](freeImageInteropNPP/)

### Description

Sample: freeImageInteropNPP
Minimum spec: SM 3.0

A simple CUDA Sample demonstrate how to use FreeImage library with NPP.

Key concepts:
Performance Strategies
Image Processing
NPP Library


## [histEqualizationNPP](histEqualizationNPP/)

### Description

Sample: histEqualizationNPP
Minimum spec: SM 3.0

This CUDA Sample demonstrates how to use NPP for histogram equalization for image data.

Key concepts:
Image Processing
Performance Strategies
NPP Library


## [jpegNPP](jpegNPP/)

### Description

Sample: jpegNPP
Minimum spec: SM 3.0

This sample demonstrates a simple image processing pipline. First, a JPEG file is huffman decoded and inverse DCT transformed and dequantized. Then the different plances are resized. Finally, the resized image is quantized, forward DCT transformed and huffman encoded.



## [MC_EstimatePiInlineP](MC_EstimatePiInlineP/)

### Description

Sample: MC_EstimatePiInlineP
Minimum spec: SM 3.0

This sample uses Monte Carlo simulation for Estimation of Pi (using inline PRNG).  This sample also uses the NVIDIA CURAND library.

Key concepts:
Random Number Generator
Computational Finance
CURAND Library


## [MC_EstimatePiInlineQ](MC_EstimatePiInlineQ/)

### Description

Sample: MC_EstimatePiInlineQ
Minimum spec: SM 3.0

This sample uses Monte Carlo simulation for Estimation of Pi (using inline QRNG).  This sample also uses the NVIDIA CURAND library.

Key concepts:
Random Number Generator
Computational Finance
CURAND Library


## [MC_EstimatePiP](MC_EstimatePiP/)

### Description

Sample: MC_EstimatePiP
Minimum spec: SM 3.0

This sample uses Monte Carlo simulation for Estimation of Pi (using batch PRNG).  This sample also uses the NVIDIA CURAND library.

Key concepts:
Random Number Generator
Computational Finance
CURAND Library


## [MC_EstimatePiQ](MC_EstimatePiQ/)

### Description

Sample: MC_EstimatePiQ
Minimum spec: SM 3.0

This sample uses Monte Carlo simulation for Estimation of Pi (using batch QRNG).  This sample also uses the NVIDIA CURAND library.

Key concepts:
Random Number Generator
Computational Finance
CURAND Library


## [MC_SingleAsianOptionP](MC_SingleAsianOptionP/)

### Description

Sample: MC_SingleAsianOptionP
Minimum spec: SM 3.0

This sample uses Monte Carlo to simulate Single Asian Options using the NVIDIA CURAND library.

Key concepts:
Random Number Generator
Computational Finance
CURAND Library


## [MersenneTwisterGP11213](MersenneTwisterGP11213/)

### Description

Sample: MersenneTwisterGP11213
Minimum spec: SM 3.0

This sample demonstrates the Mersenne Twister random number generator GP11213 in cuRAND.

Key concepts:
Computational Finance
CURAND Library


## [nvgraph_Pagerank](nvgraph_Pagerank/)

### Description

Sample: nvgraph_Pagerank
Minimum spec: SM 3.0

A CUDA Sample that demonstrates Page Rank computation using NVGRAPH Library.

Key concepts:
Graph Analytics
NVGRAPH Library


## [nvgraph_SemiRingSpMV](nvgraph_SemiRingSpMV/)

### Description

Sample: nvgraph_SemiRingSpMV
Minimum spec: SM 3.0

A CUDA Sample that demonstrates Semi-Ring SpMV using NVGRAPH Library.

Key concepts:
Graph Analytics
NVGRAPH Library


## [nvgraph_SpectralClustering](nvgraph_SpectralClustering/)

### Description

Sample: nvgraph_SpectralClustering
Minimum spec: SM 3.0

A CUDA Sample that demonstrates Spectral Clustering using NVGRAPH Library.

Key concepts:
Graph Analytics
NVGRAPH Library


## [nvgraph_SSSP](nvgraph_SSSP/)

### Description

Sample: nvgraph_SSSP
Minimum spec: SM 3.0

A CUDA Sample that demonstrates Single Source Shortest Path(SSSP) computation using NVGRAPH Library.

Key concepts:
Graph Analytics
NVGRAPH Library


## [randomFog](randomFog/)

### Description

Sample: randomFog
Minimum spec: SM 3.0

This sample illustrates pseudo- and quasi- random numbers produced by CURAND.

Key concepts:
3D Graphics
CURAND Library


## [simpleCUBLAS](simpleCUBLAS/)

### Description

Sample: simpleCUBLAS
Minimum spec: SM 3.0

Example of using CUBLAS using the new CUBLAS API interface available in CUDA 4.0.

Key concepts:
Image Processing
CUBLAS Library


## [simpleCUBLASXT](simpleCUBLASXT/)

### Description

Sample: simpleCUBLASXT
Minimum spec: SM 3.0

Example of using CUBLAS-XT library.

Key concepts:
CUBLAS-XT Library


## [simpleCUFFT](simpleCUFFT/)

### Description

Sample: simpleCUFFT
Minimum spec: SM 3.0

Example of using CUFFT. In this example, CUFFT is used to compute the 1D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain. cuFFT plans are created using simple and advanced API functions.

Key concepts:
Image Processing
CUFFT Library


## [simpleCUFFT_2d_MGPU](simpleCUFFT_2d_MGPU/)

### Description

Sample: simpleCUFFT_2d_MGPU
Minimum spec: SM 3.0

Example of using CUFFT. In this example, CUFFT is used to compute the 2D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain on Multiple GPU.

Key concepts:
Image Processing
CUFFT Library


## [simpleCUFFT_callback](simpleCUFFT_callback/)

### Description

Sample: simpleCUFFT_callback
Minimum spec: SM 3.0

Example of using CUFFT. In this example, CUFFT is used to compute the 1D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain. The difference between this example and the Simple CUFFT example is that the multiplication step is done by the CUFFT kernel with a user-supplied CUFFT callback routine, rather than by a separate kernel call.

Key concepts:
Image Processing
CUFFT Library


## [simpleCUFFT_MGPU](simpleCUFFT_MGPU/)

### Description

Sample: simpleCUFFT_MGPU
Minimum spec: SM 3.0

Example of using CUFFT. In this example, CUFFT is used to compute the 1D-convolution of some signal with some filter by transforming both into frequency domain, multiplying them together, and transforming the signal back to time domain on Multiple GPU.

Key concepts:
Image Processing
CUFFT Library


