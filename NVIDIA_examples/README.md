# NVIDIA examples

**These examples are CUDA exampes from NVIDIA, shipped with the NVIDIA CUDA TOOLKIT.**

You can find these: ```/usr/local/cuda-X.Y/samples/```. In this repository *cuda-10.0* is used.

## Compile the examples

In order to compile them use the **Makefile**, which recursively calls the Makefiles for each example.

Since the compiler and some flags has to be changed, the **Makefile** is modified, so that:

* script *replace_strings* is for every example
	* copied
	* called 
* ...

The *replace_strings* script looks like

```
cd $(dirname $0)
sed -i 's/^HOST_COMPILER ?= g++$/HOST_COMPILER ?= g++-6/gm' Makefile
sed -i 's/^NVCCFLAGS   := -m${TARGET_SIZE}$/NVCCFLAGS   := -m${TARGET_SIZE} -I\/usr\/local\/cuda-10.0\/samples\/common\/inc\//gm' Makefile
```
and can be modified in order to fullfill the computer settings.

## The NVIDIA CUDA examples

For reference see [CUDA Toolkit Documentation - CUDA samples](https://docs.nvidia.com/cuda/cuda-samples/index.html)

### [0_Simple](0_Simple/)

Basic CUDA samples illustrating key concepts of CUDA and CUDA runtime APIs.


### [1_Utilities](1_Utilities/)

Utilitity samples demonstrating how to query device capabilities and measuring GPU/CPU bandwith(s).


### [2_Graphics](2_Graphics/)

Graphical samples demonstrating interoperability between CUDA and OpenGL/DirectX. 


### [3_Imaging](3_Imaging/)

Samples demonstrating image processing, compression and data analysis.


### [4_Finance](4_Finance/)

Samples demonstrating parallel algorithms for financial computing.


### [5_Simulations](5_Simulations/)

Samples illustrating a number of simulation algorithms that uses CUDA.


### [6_Advanced](6_Advanced/)

Samples illustrating advanced algorithms implemented with CUDA.


### [7_CUDALibraries](7_CUDALibraries/)

Examples illustrating the usage of CUDA platform libraries, like:

* NPP (2D image and signal processing)
* NVJPEG (high-performance JPEG decoding)
* NVGRAPH (separates the topology of a graph from the values)
* cuBLAS (Basic Linear Algebra Subprograms)
* cuFFT (CUDA FFT library)
* cuSPARSE (et of basic linear algebra subroutines used for handling sparse matrices)
* cuSOLVER (library for decompositions and linear system solutions for both dense and sparse matrices) 
* cuRAND (random number generation library)
* ...
