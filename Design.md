Design Document: Lighter++: AC++/CUDA LLM Inference Engine
Author: Nate Willis
Date: December 2025
Status: In Development


1. Project Abstract
This project is a high-performance, custom-built Inference Engine for Large Language Models (LLMs), specifically targeting the Llama 2 architecture.  Implemented entirely in C++ and CUDA without reliance on heavy frameworks like PyTorch or TensorFlow, this system demonstrates a "from-scratch" understanding of tensor operations, memory management, and GPU kernel optimization.

The engine is designed to load raw binary model weights, manage KV-cache state for autoregressive generation, and execute the full Transformer forward pass with custom CUDA kernels optimized for latency and throughput.


2. System Architecture
The system is composed of four distinct layers:
    2.1 The Loader Layer (src/loader/)
    Responsibility: Interfacing with the disk.
    Functionality:
    Maps raw binary model files (.bin) into host memory (RAM). 
    Parses the Config header to determine model topology (layers, heads, dimensions). 
    Handles weight transfer from Host (CPU) to Device (GPU/VRAM). 
    Key Challenge: Efficient file I/O and correct pointer arithmetic for weight offsets.

    2.2 The Tensor Layer (src/tensor/)
    Responsibility: Data abstraction.
    Functionality:
    Defines the Tensor struct (shape, stride, data pointer). 
    Manages memory allocation (cudaMalloc, malloc) and deallocation. 
    Provides synchronization primitives for CPU-GPU data transfer. 

    2.3 The Operations Layer (src/ops/)
    Responsibility: The Mathematical Engine.
    Components:
    CPU Kernels: Naive implementations for verification and debugging. 
    GPU Kernels (CUDA):
    MatMul: Custom matrix multiplication with Shared Memory tiling. 
    RoPE: Rotary Positional Embeddings for attention heads. 
    RMSNorm: Root Mean Square Normalization. 
    SwiGLU: Silu-based Gated Linear Unit activation. 
    Softmax: Numerically stable probability calculation.

    2.4 The Transformer Layer (src/model/)
    Responsibility: Wiring the graph.
    Functionality:
    Implements TransformerBlock: The repeated unit of the model (Attention + FeedForward).
    Implements LlamaModel: Orchestrates the loop over all layers.
    Manages the KV-Cache (Key-Value Cache) to optimize generation speed by storing past token computations.

    2.5 The Sampler (src/sampler/)
    Responsibility: Token generation.
    Functionality:
    Accepts logits (unnormalized scores) from the model.
    Applies Temperature scaling.
    Performs Top-P (Nucleus) and Top-K sampling.
    Selects the next token index.


3. Technical Stack
Language: C++17 (Core Logic), CUDA C++ (Kernels). 
Build System: CMake (Cross-platform build generation). 
Version Control: Git. 
External Dependencies: None (Standard Library only).
Target Hardware: NVIDIA GPUs (Compute Capability 6.0+ recommended).


4. Implementation Roadmap (8 Weeks)
    Phase I: The CPU Foundation (Weeks 1-4)
        Goal: A working, strictly correct inference engine running on CPU.
        Week 1: Project skeleton, build system, Tensor class, and Model Loader.
        Week 2: Implementation of Transformer blocks (RMSNorm, MatMul, SwiGLU).
        Week 3: Attention Mechanism (Multi-Head Attention, RoPE).
        Week 4: Tokenizer, Sampler, and full end-to-end inference verification.

    Phase II: The GPU Acceleration (Weeks 5-7)
        Goal: Porting operations to CUDA and optimizing for speed.
        Week 5: CUDA Context setup, memory transfer, and Pointwise kernels (Add, Norm).
        Week 6: Matrix Multiplication Optimization. Implementing tiled SGEMM kernels using shared memory.
        Week 7: KV-Cache implementation and Kernel Fusion (merging operations to reduce memory bandwidth).

    Phase III: Polish & Benchmarking (Week 8)
        Goal: Documentation and Performance analysis.
        Week 8:
        Profiling using NVIDIA Nsight Systems. 
        Comparison: Custom Engine vs llama.cpp performance.
        Final code cleanup and documentation. 

5. Future Work: The Training Engine
Upon completion of the Inference Engine, the project will be extended to include a Training Engine.
Scope: Implementation of Backpropagation (Autodiff) for the Llama architecture.
Dataset: roneneldan/TinyStories (Public HuggingFace Dataset). 
Goal: Train a 15M parameter model from scratch to verify the full ML lifecycle.