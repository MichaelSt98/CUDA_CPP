# 1_Utilities

## [bandwidthTest](bandwidthTest/)

### Description

Sample: bandwidthTest
Minimum spec: SM 3.0

This is a simple test program to measure the memcopy bandwidth of the GPU and memcpy bandwidth across PCI-e.  This test application is capable of measuring device to device copy bandwidth, host to device copy bandwidth for pageable and page-locked memory, and device to host copy bandwidth for pageable and page-locked memory.

Key concepts:
CUDA Streams and Events
Performance Strategies


## [deviceQuery](deviceQuery/)

### Description

Sample: deviceQuery
Minimum spec: SM 3.0

This sample enumerates the properties of the CUDA devices present in the system.

Key concepts:
CUDA Runtime API
Device Query


## [deviceQueryDrv](deviceQueryDrv/)

### Description

Sample: deviceQueryDrv
Minimum spec: SM 3.0

This sample enumerates the properties of the CUDA devices present using CUDA Driver API calls

Key concepts:
CUDA Driver API
Device Query


## [p2pBandwidthLatencyTest](p2pBandwidthLatencyTest/)

### Description

Sample: p2pBandwidthLatencyTest
Minimum spec: SM 3.0

This application demonstrates the CUDA Peer-To-Peer (P2P) data transfers between pairs of GPUs and computes latency and bandwidth.  Tests on GPU pairs using P2P and without P2P are tested.

Key concepts:
Performance Strategies
Asynchronous Data Transfers
Unified Virtual Address Space
Peer to Peer Data Transfers
Multi-GPU


## [topologyQuery](topologyQuery/)

### Description

Sample: topologyQuery
Minimum spec: SM 3.0

A simple exemple on how to query the topology of a system with multiple GPU

Key concepts:
Performance Strategies
Multi-GPU


## [UnifiedMemoryPerf](UnifiedMemoryPerf/)

### Description

Sample: UnifiedMemoryPerf
Minimum spec: SM 3.0

This sample demonstrates the performance comparision using matrix multiplication kernel of Unified Memory with/without hints and other types of memory like zero copy buffers, pageable, pagelocked memory performing synchronous and Asynchronous transfers on a single GPU.

Key concepts:
CUDA Systems Integration
Unified Memory
CUDA Streams and Events
Pinned System Paged Memory


