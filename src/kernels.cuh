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

// CUDA Scale
// Multiplies every element by a scalar: x[i] = x[i] * scale
// Parameters:
//   d_x:   Vector to scale in-place
//   scale: Scalar multiplier
//   n:     Number of elements
void cuda_scale(float* d_x, float scale, int n);

// CUDA Scale (multi-head) - scales all attention heads in one launch
// Parameters:
//   d_att:      Attention scores [n_heads × att_stride]
//   scale:      Scalar multiplier (typically 1/sqrt(head_size))
//   n_heads:    Number of attention heads
//   seq_len:    Actual positions to scale (pos + 1)
//   att_stride: Stride between heads (p->seq_len)
void cuda_scale_multihead(float* d_att, float scale, int n_heads, int seq_len, int att_stride);

// CUDA Residual Add
// Computes element-wise addition: out = a + b
// Parameters:
//   d_out: Output vector (can be same as d_a for in-place)
//   d_a:   First input vector
//   d_b:   Second input vector
//   n:     Number of elements
void cuda_residual_add(float* d_out, const float* d_a, const float* d_b, int n);

// CUDA Softmax
// Computes numerically-stable softmax: out[i] = exp(x[i] - max) / sum(exp(x[j] - max))
// Parameters:
//   d_out: Output vector (size n) - probabilities summing to 1
//   d_x:   Input vector (size n) - logits
//   n:     Number of elements
void cuda_softmax(float* d_out, float* d_x, int n);

// CUDA Softmax (multi-head) - softmax all attention heads in one launch
// Parameters:
//   d_out:      Output [n_heads × att_stride]
//   d_x:        Input [n_heads × att_stride]
//   n_heads:    Number of attention heads
//   seq_len:    Actual positions to process (pos + 1)
//   att_stride: Stride between heads (p->seq_len)
void cuda_softmax_multihead(float* d_out, float* d_x, int n_heads, int seq_len, int att_stride);

// CUDA Aggregation (multi-head) - processes all heads in one launch
// Parameters:
//   d_out:      Output [n_heads × head_size] (same as dim)
//   d_v:        Value cache [n_kv_heads × seq_len × head_size] (with layer offset applied)
//   d_att:      Attention scores [n_heads × seq_len]
//   n_heads:    Number of query heads
//   seq_len:    Number of positions (pos + 1)
//   head_size:  Dimension per head
//   gqa_factor: n_heads / n_kv_heads (for grouped query attention)
//   att_stride: Stride between heads in attention buffer (p->seq_len)
void cuda_aggregation_multihead(float* d_out, const float* d_v, const float* d_att, int n_heads, int seq_len, int head_size, int gqa_factor, int att_stride);

#endif
