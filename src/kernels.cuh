#ifndef KERNELS_CUH
#define KERNELS_CUH

// CUDA GEMV (General Matrix-Vector multiply)
// Computes: out = W * x
// Parameters:
//   d_out: Output vector (size n)
//   d_x:   Input vector (size d)  
//   d_w:   Weight matrix (n rows, d columns) - flattened, row-major
//   n:   Number of rows (output dimension)
//   d:   Number of columns (input dimension)
void cuda_gemv(float* d_out, float* d_x, float* d_w, int n, int d);

// CUDA RMSNorm
// Computes: out = x / sqrt(var + epsilon) * w
// Parameters:
//   d_out: Output vector (size n)
//   d_x:   Input vector (size n)
//   d_w:   Weight vector (size n)
//   n:   Number of elements
void cuda_rmsnorm(float* d_out, float* d_x, float* d_w, int n);

// CUDA RoPE (Rotary Position Embedding)
// Applies rotation to Q and K vectors based on position
// Parameters:
//   d_q:       Query vector (size dim), modified in-place
//   d_k:       Key vector (size kv_dim), modified in-place
//   pos:       Token position in sequence
//   dim:       Query dimension
//   kv_dim:    Key/Value dimension
//   head_size: Size of each attention head
void cuda_rope(float* d_q, float* d_k, int pos, int dim, int kv_dim, int head_size);

// CUDA SwiGLU
// Computes: hb = SiLU(h1) * h3
// Parameters:
//   d_hb: Output vector (size n)
//   d_h1: Gate input vector (size n)
//   d_h3: Value input vector (size n)
//   size: Number of elements
void cuda_swiglu(float* d_hb, float* d_h1, float* d_h3, int size);

#endif
