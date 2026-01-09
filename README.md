![Build Status](https://github.com/21NWillis/Lighterpp/actions/workflows/ci.yml/badge.svg)

# Lighter++: A C++/CUDA LLM Inference Engine

**Author:** Nate Willis  
**Status:** In Development (Phase I: CPU Foundation: Completed, Phase II: Transformer Architecture: Completed, Phase III: CUDA Acceleration: In Progress)


![Lighter++ Demo](assets/demo.gif)
*(Running inference on the TinyStories 15M parameter model)*

## Abstract
Lighter++ is a high-performance, custom-built Inference Engine for Large Language Models (LLMs), specifically targeting the Llama 2 architecture. Implemented entirely in C++ (and eventually CUDA) without using existing frameworks like PyTorch or TensorFlow. This is to learn and demonstrate a "from-scratch" understanding of tensor operations, memory management, and hardware optimization.

The engine currently supports loading `llama2.c` compatible model checkpoints (e.g., `stories15M.bin`).


### Performance Benchmarks
**Hardware:** Intel i7-1165G7 Laptop CPU (Inference run on `stories15M.bin`)

| Engine | Performance (tok/s) |
| :--- | :--- |
| **Lighter++** (Current) | 56.60 |
| **llama2.c** (Reference) | 72.40 |

### Why is Lighter++ slower than llama2.c?
Lighter++ is currently built for **clarity and correctness** over raw CPU optimization. The reference implementation `llama2.c` optimizes for SIMD vectorization and other micro-optimizations, while Lighter++ uses a more straightforward approach with the intent to port the operations to CUDA. I chose not to focus my time optimizing the CPU code when I know I will be focusing more on the GPU code in the future.

## Key Architectural Decisions

*   **Standard Library Only:** Built with zero external dependencies to ensure a "from-scratch" understanding of the math and memory management.
*   **Memory Mapped Loading (`mmap`):** Instead of using standard file I/O, model weights are mapped directly into the process's virtual address space. This is a "zero-copy" implementation of loading, using "lazy loading" to only load the weights as they are needed.
*   **Flattened Tensor Layout:** The KV Cache specifically uses a strided memory layout to ensure that attention keys and values are contiguous in memory, maximizing CPU cache hits during the Attention step. It sacrifices write time performance, with several cache misses during the writing, but optimizes the read time performance, which occurs much more frequently in the inference loop.

## Difficult Issues & Bug Log

### 1. The "Six Layer 1s" Bug
Early in Phase II, the model was generating tokens but was collapsing into repetition after only the first few tokens were generated. This was caused by the layer 1 weights being used for every layer in the transformer loop. I continued to build on top of this for a while before discovering it, so when I finally DID, it was a mess to figure out where things were going wrong. In the end, I found the issue by "rubber duck debugging" and realizing that I was not adding a layer offset to the attention function.
*   **Cause:** A missing layer offset in the attention function, causing the first layer's weights to be used for every layer.
*   **Fix:** Implemented specific `layer_offset` logic for the attention function.

### 2. The "Destructive In-place MatMul" Bug
A subtle bug caused the model to fall into nonsensical loops (e.g., "flowers to shower the flowers"). This required me to verify the output of the attention function against the reference implementation to find the source of the issue.
*   **Cause:** The Attention output projection was using the same buffer (xb) for its source and its output. This was causing destructive in-place matrix multiplication that ruined the weight calculations.
*   **Fix:** Implemented separate buffers (xb, xb2) in `RunState` to prevent the buffer from being overwritten while in use.

**Lesson Learned:** Comprehensive unit tests are necessary to catch silent failures and reduce the amount of time spent aimlessly searching for the source of a bug.


## Technical Stack
*   **Language:** C++17
*   **Build System:** CMake
*   **External Dependencies:** None (Standard Library only)
*   **Target Architecture:** CPU (Phase I), NVIDIA GPU/CUDA (Phase II)

## Project Structure

```text
├── src/
│   ├── loader.h/cpp     # Binary file I/O and memory mapping
│   ├── model.h/cpp      # Model topology (Config) and weight pointers
│   ├── tensor.h/cpp     # Tensor structures and data definitions
│   ├── ops.h/cpp        # Math kernels (MatMul, RMSNorm, RoPE)
│   ├── tokenizer.h/cpp  # Tokenizer
│   ├── main.cpp         # Application entry point
│   └── runtests.cpp     # Unit testing suite (math functions)
├── CMakeLists.txt       # Build configuration
└── README.md            # Documentation
```

## Getting Started
### 1. Build
Lighter++ uses CMake. Ensure you have a C++17 compiler installed.

```bash
mkdir build
cd build
cmake ..
make
```

### 2. Download Model & Tokenizer
This engine is compatible with the "stories" models from the TinyLlamas project.

You can download the 15M parameter model and its tokenizer using the following commands:

```bash
# Download Model
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# Download Tokenizer
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

Ensure both files are in the Lighterpp directory.

### 3. Run Verification
Lighter++ includes a test suite to verify the mathematical correctness of its operations.

```bash
./runtests
```

### 4. Run the Engine
Load the model to verify weight parsing and architecture configuration.

Use this command from inside the build directory:

```bash
./Lighter++ ../stories15M.bin ../tokenizer.bin [temperature]
```
*   `temperature`: (Optional) Randomness of generation. `0.0` for deterministic, `0.9` default.


## Implementation Roadmap

### Phase I: The CPU Foundation (Completed)

- [x] Project Setup: CMake build system and repo structure.
- [x] Tensor Core: Data structures for tensor operations.
- [x] Model Loader: mmap based binary weight loading.
- [x] Basic Math: Naive Matrix-Vector multiplication (GEMV).
- [x] Testing: Unit test suite for linear algebra operations.

### Phase II: The Transformer Architecture (Completed)

- [x] State Management: KV Cache allocation and management.
- [x] Normalization: RMSNorm implementation.
- [x] Positional Embeddings: Rotary Positional Embeddings (RoPE).
- [x] Activation Functions: Softmax and SwiGLU.
- [x] Attention: Multi-Head Attention with Grouped Query Attention (GQA) support.
- [x] FeedForward Network: Gate, Up, and Down projections with SwiGLU.
- [x] Transformer Block: Full layer with residual connections.
- [x] Single-Token Inference: Forward pass through all layers.
- [x] Multi-Token Generation: Autoregressive sampling loop.
- [x] Tokenizer: Decode token IDs to text.

### Phase III: CUDA Acceleration (In Progress)

- [ ] GPU Memory Management.
- [ ] Custom CUDA Kernels for MatMul and Attention.
- [ ] Profiling and Optimization.

### Phase IV: Final Cleanup (Future)
- [ ] Final Cleanup and Documentation.
- [ ] Final Testing and Validation.
- [ ] Final Deployment and Release.


## Acknowledgments

*   **Andrej Karpathy:** For [llama2.c](https://github.com/karpathy/llama2.c), which served as the direct reference implementation and inspiration for this project.
*   **Meta AI:** For the design of the original [Llama 2 architecture](https://ai.meta.com/llama/), which defines the mathematical structure used in this engine.
*   **TinyStories:** [TinyStories: How Small Can Language Models Be?](https://arxiv.org/abs/2305.07759) - The paper and source for the models used in testing.