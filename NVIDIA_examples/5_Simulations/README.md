# 5_Simulations

## [fluidsGL](fluidsGL/)

### Description

Sample: fluidsGL
Minimum spec: SM 3.0

An example of fluid simulation using CUDA and CUFFT, with OpenGL rendering.

Key concepts:
Graphics Interop
CUFFT Library
Physically-Based Simulation


## [fluidsGLES](fluidsGLES/)

### Description

Sample: fluidsGLES
Minimum spec: SM 3.0

An example of fluid simulation using CUDA and CUFFT, with OpenGLES rendering.

Key concepts:
Graphics Interop
CUFFT Library
Physically-Based Simulation


## [nbody](nbody/)

### Description

Sample: nbody
Minimum spec: SM 3.0

This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA.  This sample accompanies the GPU Gems 3 chapter "Fast N-Body Simulation with CUDA".  With CUDA 5.5, performance on Tesla K20c has increased to over 1.8TFLOP/s single precision.  Double Performance has also improved on all Kepler and Fermi GPU architectures as well.  Starting in CUDA 4.0, the nBody sample has been updated to take advantage of new features to easily scale the n-body simulation across multiple GPUs in a single PC.  Adding "-numbodies=<bodies>" to the command line will allow users to set # of bodies for simulation.  Adding “-numdevices=<N>” to the command line option will cause the sample to use N devices (if available) for simulation.  In this mode, the position and velocity data for all bodies are read from system memory using “zero copy” rather than from device memory.  For a small number of devices (4 or fewer) and a large enough number of bodies, bandwidth is not a bottleneck so we can achieve strong scaling across these devices.

Key concepts:
Graphics Interop
Data Parallel Algorithms
Physically-Based Simulation


## [nbody_opengles](nbody_opengles/)

### Description

Sample: nbody_opengles
Minimum spec: SM 3.0

This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA. Unlike the OpenGL nbody sample, there is no user interaction.

Key concepts:
Graphics Interop
Data Parallel Algorithms
Physically-Based Simulation


## [nbody_screen](nbody_screen/)

### Description

Sample: nbody_screen
Minimum spec: SM 3.0

This sample demonstrates efficient all-pairs simulation of a gravitational n-body simulation in CUDA. Unlike the OpenGL nbody sample, there is no user interaction.

Key concepts:
Graphics Interop
Data Parallel Algorithms
Physically-Based Simulation


## [oceanFFT](oceanFFT/)

### Description

Sample: oceanFFT
Minimum spec: SM 3.0

This sample simulates an Ocean height field using CUFFT Library and renders the result using OpenGL.

Key concepts:
Graphics Interop
Image Processing
CUFFT Library


## [particles](particles/)

### Description

Sample: particles
Minimum spec: SM 3.0

This sample uses CUDA to simulate and visualize a large set of particles and their physical interaction.  Adding "-particles=<N>" to the command line will allow users to set # of particles for simulation.  This example implements a uniform grid data structure using either atomic operations or a fast radix sort from the Thrust library

Key concepts:
Graphics Interop
Data Parallel Algorithms
Physically-Based Simulation
Performance Strategies


## [smokeParticles](smokeParticles/)

### Description

Sample: smokeParticles
Minimum spec: SM 3.0

Smoke simulation with volumetric shadows using half-angle slicing technique. Uses CUDA for procedural simulation, Thrust Library for sorting algorithms, and OpenGL for graphics rendering.

Key concepts:
Graphics Interop
Data Parallel Algorithms
Physically-Based Simulation


