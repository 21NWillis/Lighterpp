![Build Status](https://github.com/21NWillis/Lighterpp/actions/workflows/ci.yml/badge.svg)

# Lighter++: A C++/CUDA LLM Inference Engine

**Author:** Nate Willis  
**Status:** In Development (Phase I: CPU Foundation: Completed, Phase II: Transformer Architecture: Completed, Phase III: CUDA Acceleration: In Progress)


![Lighter++ Demo](assets/demo.gif)
*(Running inference on the TinyStories 42M parameter model)*

## Abstract
Lighter++ is a high-performance, custom-built Inference Engine for Large Language Models (LLMs), specifically targeting the Llama 2 architecture. Implemented entirely in C++ (and eventually CUDA) without using existing frameworks like PyTorch or TensorFlow. This is to learn and demonstrate a "from-scratch" understanding of tensor operations, memory management, and hardware optimization.

The engine currently supports loading `llama2.c` compatible model checkpoints (e.g., `stories15M.bin`).

## Technical Stack
*   **Language:** C++17
*   **Build System:** CMake
*   **External Dependencies:** None (Standard Library only)
*   **Target Architecture:** CPU (Phase I), NVIDIA GPU/CUDA (Phase II)

## Project Structure

```text
├── src/
│   ├── loader/     # Binary file I/O and memory mapping
│   ├── model/      # Model topology (Config) and weight pointers
│   ├── tensor/     # Tensor structures and data definitions
│   ├── ops/        # Math kernels (MatMul, RMSNorm, RoPE)
│   ├── tokenizer.cpp/h # Tokenizer implementation
│   ├── main.cpp    # Application entry point
│   └── runtests.cpp   # Unit testing suite (math functions)
├── CMakeLists.txt  # Build configuration
└── README.md       # Documentation
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