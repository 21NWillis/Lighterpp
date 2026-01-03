# Lighter++: A C++/CUDA LLM Inference Engine

**Author:** Nate Willis  
**Status:** In Development (Phase I: CPU Foundation: Completed, Phase II: Transformer Architecture: In Progress)

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
│   ├── main.cpp    # Application entry point
│   └── tests.cpp   # Unit testing suite
├── CMakeLists.txt  # Build configuration
└── README.md       # Documentation
```

## Getting Started
1. Build
Lighter++ uses CMake. Ensure you have a C++17 compiler installed.

```bash
mkdir build
cd build
cmake ..
make
```

2. Download Model
This engine is compatible with the "stories" models from the TinyLlamas project.

You can download the 15M parameter model (~60MB) using the following command:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Ensure the model is in the Lighterpp directory.


3. Run Verification
Lighter++ includes a test suite to verify the mathematical correctness of its operations.

```bash
./runtests
```


4. Run the Engine
Load the model to verify weight parsing and architecture configuration.

Use this command from inside the build directory:

```bash
./Lighter++ ../stories15M.bin
```


## Implementation Roadmap

### Phase I: The CPU Foundation (Completed)

- [x] Project Setup: CMake build system and repo structure.
- [x] Tensor Core: Data structures for tensor operations.
- [x] Model Loader: mmap based binary weight loading.
- [x] Basic Math: Naive Matrix-Vector multiplication (GEMV).
- [x] Testing: Unit test suite for linear algebra operations.

### Phase II: The Transformer Architecture (In Progress)

- [ ] State Management: KV Cache allocation and management.
- [ ] Normalization: RMSNorm implementation.
- [ ] Positional Embeddings: Rotary Positional Embeddings (RoPE).
- [ ] Attention: Multi-Head Attention (MHA) logic.
- [ ] Inference Loop: Full autoregressive generation.

### Phase III: CUDA Acceleration (Future)

- [ ] GPU Memory Management.
- [ ] Custom CUDA Kernels for MatMul and Attention.
- [ ] Profiling and Optimization.

### Phase IV: Final Cleanup (Future)
- [ ] Final Cleanup and Documentation.
- [ ] Final Testing and Validation.
- [ ] Final Deployment and Release.

