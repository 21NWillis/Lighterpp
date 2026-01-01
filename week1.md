Week 1: Setup, Tensors, and The Loader


Day 1: The Skeleton // DONE//
Time Expectation: 1.5 Hours
Goal: Initialize the repo and get a clean build system running. No code is better than code that doesn't compile.

Repo: Create llama-scratch, git init.

Ignore: Create .gitignore (add build/, .vscode/, *.bin, *.exe).

CMake: Create CMakeLists.txt.
Required: set(CMAKE_CXX_STANDARD 17)
Flags: -O3 -Wall -Wextra (We want the compiler to yell at us).

Entry: Create src/main.cpp.
Print "Inference Engine Initialized" and sizeof(float).

Build: Run mkdir build && cd build && cmake .. && make (or equivalent).
🛑 Commit Before Bed:
"Initial build system setup. Hello World prints successfully."


Day 2: The Tensor Class //DONE//
Time Expectation: 2 - 2.5 Hours
Goal: Create the data structure that will hold everything.

Header: Create src/tensor.h.

Struct: Define struct Tensor.
code
C++
struct Tensor {
    float* data;
    int shape[4]; // [Batch, Heads, Sequence, Dim]
    int size;     // Total elements
};

Methods: Implement Tensor* allocate_tensor(int shape[]) using calloc (zero-init).

Cleanup: Implement void free_tensor(Tensor* t).

CRITICAL TASK: Implement void print_tensor(Tensor* t, int n_items).
It should print the first n items nicely. You will use this to debug forever.
🛑 Commit Before Bed:
"Implemented Tensor struct, allocation logic, and debug printing."
Day 3: The Header Parser
Time Expectation: 2 Hours
Goal: Read the configuration from the binary file.

Download: Get stories15M.bin from Karpathy's HF Repo.

Config: Create src/model.h with this exact struct (must match the binary format):
code
C++
struct Config {
    int dim;        // Transformer dimension
    int hidden_dim; // FFN dimension
    int n_layers;   // Number of layers
    int n_heads;    // Number of query heads
    int n_kv_heads; // Number of key/value heads (for GQA)
    int vocab_size; // Vocabulary size
    int seq_len;    // Max sequence length
};

Reader: In main.cpp, use fopen and fread to read just the first sizeof(Config) bytes.

Verification: Print the values.
Success Criteria: dim = 288, n_layers = 6, vocab_size = 32000.
Fail Criteria: Random garbage numbers (means endianness issue or wrong struct order).
🛑 Commit Before Bed:
"Successfully parsed model header configuration from binary file."
Day 4: The Weight Loader (Pointer Math)
Time Expectation: 3 Hours
Goal: Map the rest of the file into memory and assign pointers.

Strategy: Decide on mmap (Advanced/Cleaner) or malloc + fread (Simple).
Rec: If on Linux/Mac, try mmap. If Windows, malloc the whole file size is fine for 60MB.

The Weights Struct: In src/model.h, define struct TransformerWeights.
Needs Tensor pointers for: token_embedding_table, rms_att_weight, wq, wk, wv, wo, etc.

The Checkpoint: Write the code to populate these pointers.
Hint: The file layout is: [Config Bytes] [Weights Data...].
You need to increment your pointer by tensor_size * sizeof(float) after every assignment.

Verify: Print the first 10 floats of token_embedding_table.
🛑 Commit Before Bed:
"Model loader maps all weight pointers. Verified first 10 weights match Python reference."
Day 5: The First Math (CPU)
Time Expectation: 2.5 Hours
Goal: Prove we can do math.

The Kernel: Create src/ops.cpp.

MatMul: Implement void matmul(float* out, float* x, float* w, int n, int d).
Use 3 nested for loops. Do not try to be fancy. O(N^3) is fine today.

The Test: Create a separate test function in main.cpp.
Create Matrix A (2x2) with values {1, 2, 3, 4}.
Create Matrix B (2x2) with values {1, 0, 0, 1} (Identity).
Run MatMul. Result should be {1, 2, 3, 4}.

Run it on Weights: Try multiplying a dummy vector by the token_embedding_table (just the first row). Ensure no Segfaults.
🛑 Commit Before Bed:
"Implemented naive CPU MatMul and added unit tests."
Day 6: Documentation & Architecture
Time Expectation: 2 Hours
Goal: Understand what you are building.

Visual: Draw the Llama 2 architecture on a piece of paper.
Identify: RMSNorm -> Attention -> Add -> RMSNorm -> FeedForward -> Add.

Cleanup: Refactor main.cpp. Move the loading logic into src/loader.cpp. Keep main clean.

Readme: Update README.md with build instructions and where to download the model.
🛑 Commit Before Bed:
"Refactored code structure and updated documentation."
Day 7: Buffer / Rest
Time Expectation: 0 Hours (or Catch-up)
Goal: Mental Reset.
If Day 4 (Pointer Math) kicked your butt, finish it today.
If you are done, close the laptop. Next week we build the Transformer Layers.