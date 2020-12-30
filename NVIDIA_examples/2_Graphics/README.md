# 2_Graphics

## [bindlessTexture](bindlessTexture/)

### Description

Sample: bindlessTexture
Minimum spec: SM 3.0

This example demonstrates use of cudaSurfaceObject, cudaTextureObject, and MipMap support in CUDA.  A GPU with Compute Capability SM 3.0 is required to run the sample.

Key concepts:
Graphics Interop
Texture


## [Mandelbrot](Mandelbrot/)

### Description

Sample: Mandelbrot
Minimum spec: SM 3.0

This sample uses CUDA to compute and display the Mandelbrot or Julia sets interactively. It also illustrates the use of "double single" arithmetic to improve precision when zooming a long way into the pattern. This sample uses double precision.  Thanks to Mark Granger of NewTek who submitted this code sample.!

Key concepts:
Graphics Interop
Data Parallel Algorithms


## [marchingCubes](marchingCubes/)

### Description

Sample: marchingCubes
Minimum spec: SM 3.0

This sample extracts a geometric isosurface from a volume dataset using the marching cubes algorithm. It uses the scan (prefix sum) function from the Thrust library to perform stream compaction.

Key concepts:
OpenGL Graphics Interop
Vertex Buffers
3D Graphics
Physically Based Simulation


## [simpleGL](simpleGL/)

### Description

Sample: simpleGL
Minimum spec: SM 3.0

Simple program which demonstrates interoperability between CUDA and OpenGL. The program modifies vertex positions with CUDA and uses OpenGL to render the geometry.

Key concepts:
Graphics Interop
Vertex Buffers
3D Graphics


## [simpleGLES](simpleGLES/)

### Description

Sample: simpleGLES
Minimum spec: SM 3.0

Demonstrates data exchange between CUDA and OpenGL ES (aka Graphics interop). The program modifies vertex positions with CUDA and uses OpenGL ES to render the geometry.

Key concepts:
Graphics Interop
Vertex Buffers
3D Graphics


## [simpleGLES_EGLOutput](simpleGLES_EGLOutput/)

### Description

Sample: simpleGLES_EGLOutput
Minimum spec: SM 3.0

Demonstrates data exchange between CUDA and OpenGL ES (aka Graphics interop). The program modifies vertex positions with CUDA and uses OpenGL ES to render the geometry, and shows how to render directly to the display using the EGLOutput mechanism and the DRM library.

Key concepts:
Graphics Interop
Vertex Buffers
3D Graphics


## [simpleGLES_screen](simpleGLES_screen/)

### Description

Sample: simpleGLES_screen
Minimum spec: SM 3.0

Demonstrates data exchange between CUDA and OpenGL ES (aka Graphics interop). The program modifies vertex positions with CUDA and uses OpenGL ES to render the geometry.

Key concepts:
Graphics Interop
Vertex Buffers
3D Graphics


## [simpleTexture3D](simpleTexture3D/)

### Description

Sample: simpleTexture3D
Minimum spec: SM 3.0

Simple example that demonstrates use of 3D Textures in CUDA.

Key concepts:
Graphics Interop
Image Processing
3D Textures
Surface Writes


## [simpleVulkan](simpleVulkan/)

### Description

Sample: simpleVulkan
Minimum spec: SM 3.0

This sample demonstrates Vulkan CUDA Interop. CUDA imports the Vulkan vertex buffer and operates on it to create sinewave, and synchronizes with Vulkan through vulkan semaphores imported by CUDA. This sample depends on Vulkan SDK, GLFW3 libraries, for building this sample please refer to "Build_instructions.txt" provided in this sample's directory

Key concepts:
Graphics Interop
CUDA Vulkan Interop
Data Parallel Algorithms


## [volumeFiltering](volumeFiltering/)

### Description

Sample: volumeFiltering
Minimum spec: SM 3.0

This sample demonstrates 3D Volumetric Filtering using 3D Textures and 3D Surface Writes.

Key concepts:
Graphics Interop
Image Processing
3D Textures
Surface Writes


## [volumeRender](volumeRender/)

### Description

Sample: volumeRender
Minimum spec: SM 3.0

This sample demonstrates basic volume rendering using 3D Textures.

Key concepts:
Graphics Interop
Image Processing
3D Textures


