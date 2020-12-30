# 3_Imaging

## [bicubicTexture](bicubicTexture/)

### Description

Sample: bicubicTexture
Minimum spec: SM 3.0

This sample demonstrates how to efficiently implement a Bicubic B-spline interpolation filter with CUDA texture.

Key concepts:
Graphics Interop
Image Processing


## [bilateralFilter](bilateralFilter/)

### Description

Sample: bilateralFilter
Minimum spec: SM 3.0

Bilateral filter is an edge-preserving non-linear smoothing filter that is implemented with CUDA with OpenGL rendering. It can be used in image recovery and denoising. Each pixel is weight by considering both the spatial distance and color distance between its neighbors. Reference:"C. Tomasi, R. Manduchi, Bilateral Filtering for Gray and Color Images, proceeding of the ICCV, 1998, http://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf"

Key concepts:
Graphics Interop
Image Processing


## [boxFilter](boxFilter/)

### Description

Sample: boxFilter
Minimum spec: SM 3.0

Fast image box filter using CUDA with OpenGL rendering.

Key concepts:
Graphics Interop
Image Processing


## [convolutionFFT2D](convolutionFFT2D/)

### Description

Sample: convolutionFFT2D
Minimum spec: SM 3.0

This sample demonstrates how 2D convolutions with very large kernel sizes can be efficiently implemented using FFT transformations.

Key concepts:
Image Processing
CUFFT Library


## [convolutionSeparable](convolutionSeparable/)

### Description

Sample: convolutionSeparable
Minimum spec: SM 3.0

This sample implements a separable convolution filter of a 2D signal with a gaussian kernel.

Key concepts:
Image Processing
Data Parallel Algorithms


## [convolutionTexture](convolutionTexture/)

### Description

Sample: convolutionTexture
Minimum spec: SM 3.0

Texture-based implementation of a separable 2D convolution with a gaussian kernel. Used for performance comparison against convolutionSeparable.

Key concepts:
Image Processing
Texture
Data Parallel Algorithms


## [dct8x8](dct8x8/)

### Description

Sample: dct8x8
Minimum spec: SM 3.0

This sample demonstrates how Discrete Cosine Transform (DCT) for blocks of 8 by 8 pixels can be performed using CUDA: a naive implementation by definition and a more traditional approach used in many libraries. As opposed to implementing DCT in a fragment shader, CUDA allows for an easier and more efficient implementation.

Key concepts:
Image Processing
Video Compression


## [dwtHaar1D](dwtHaar1D/)

### Description

Sample: dwtHaar1D
Minimum spec: SM 3.0

Discrete Haar wavelet decomposition for 1D signals with a length which is a power of 2.

Key concepts:
Image Processing
Video Compression


## [dxtc](dxtc/)

### Description

Sample: dxtc
Minimum spec: SM 3.0

High Quality DXT Compression using CUDA. This example shows how to implement an existing computationally-intensive CPU compression algorithm in parallel on the GPU, and obtain an order of magnitude performance improvement.

Key concepts:
Cooperative Groups
Image Processing
Image Compression


## [EGLStream_CUDA_CrossGPU](EGLStream_CUDA_CrossGPU/)

### Description

Sample: EGLStream_CUDA_CrossGPU
Minimum spec: SM 3.0

Demonstrates CUDA and EGL Streams interop, where consumer's EGL Stream is on one GPU and producer's on other and both consumer-producer are different processes.

Key concepts:
EGLStreams Interop


## [EGLStreams_CUDA_Interop](EGLStreams_CUDA_Interop/)

### Description

Sample: EGLStreams_CUDA_Interop
Minimum spec: SM 3.0

Demonstrates data exchange between CUDA and EGL Streams.

Key concepts:
EGLStreams Interop


## [EGLSync_CUDAEvent_Interop](EGLSync_CUDAEvent_Interop/)

### Description

Sample: EGLSync_CUDAEvent_Interop
Minimum spec: SM 3.0

Demonstrates interoperability between CUDA Event and EGL Sync/EGL Image using which one can achieve synchronization on GPU itself for GL-EGL-CUDA operations instead of blocking CPU for synchronization.

Key concepts:
EGLSync-CUDAEvent Interop
EGLImage-CUDA Interop


## [histogram](histogram/)

### Description

Sample: histogram
Minimum spec: SM 3.0

This sample demonstrates efficient implementation of 64-bin and 256-bin histogram.

Key concepts:
Image Processing
Data Parallel Algorithms


## [HSOpticalFlow](HSOpticalFlow/)

### Description

Sample: HSOpticalFlow
Minimum spec: SM 3.0

Variational optical flow estimation example.  Uses textures for image operations. Shows how simple PDE solver can be accelerated with CUDA.

Key concepts:
Image Processing
Data Parallel Algorithms


## [imageDenoising](imageDenoising/)

### Description

Sample: imageDenoising
Minimum spec: SM 3.0

This sample demonstrates two adaptive image denoising techniques: KNN and NLM, based on computation of both geometric and color distance between texels. While both techniques are implemented in the DirectX SDK using shaders, massively speeded up variation of the latter technique, taking advantage of shared memory, is implemented in addition to DirectX counterparts.

Key concepts:
Image Processing


## [postProcessGL](postProcessGL/)

### Description

Sample: postProcessGL
Minimum spec: SM 3.0

This sample shows how to post-process an image rendered in OpenGL using CUDA.

Key concepts:
Graphics Interop
Image Processing


## [recursiveGaussian](recursiveGaussian/)

### Description

Sample: recursiveGaussian
Minimum spec: SM 3.0

This sample implements a Gaussian blur using Deriche's recursive method. The advantage of this method is that the execution time is independent of the filter width.

Key concepts:
Graphics Interop
Image Processing


## [simpleCUDA2GL](simpleCUDA2GL/)

### Description

Sample: simpleCUDA2GL
Minimum spec: SM 3.0

This sample shows how to copy CUDA image back to OpenGL using the most efficient methods.

Key concepts:
Graphics Interop
Image Processing
Performance Strategies


## [SobelFilter](SobelFilter/)

### Description

Sample: SobelFilter
Minimum spec: SM 3.0

This sample implements the Sobel edge detection filter for 8-bit monochrome images.

Key concepts:
Graphics Interop
Image Processing


## [stereoDisparity](stereoDisparity/)

### Description

Sample: stereoDisparity
Minimum spec: SM 3.0

A CUDA program that demonstrates how to compute a stereo disparity map using SIMD SAD (Sum of Absolute Difference) intrinsics.  Requires Compute Capability 2.0 or higher.

Key concepts:
Image Processing
Video Intrinsics


