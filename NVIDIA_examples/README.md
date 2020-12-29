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

### [0_Simple](0_Simple/)


### [1_Utilities](1_Utilities/)


### [2_Graphics](2_Graphics/)


### [3_Imaging](3_Imaging/)


### [4_Finance](4_Finance/)


### [5_Simulations](5_Simulations/)


### [6_Advanced](6_Advanced/)


### [7_CUDALibraries](7_CUDALibraries/)
