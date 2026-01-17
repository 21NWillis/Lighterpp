#ifndef OPS_H
#define OPS_H

// Naive Matrix Multiplication
// C = A * B
// Parameters:
//  out: Output vector (size n)
//  x:   Input vector (size d)
//  w:   Weight matrix (n rows, d columns) - flattened, row-major
//  n:   Output dimension (rows of w)
//  d:   Input dimension (size of x, columns of w)
void naive_matmul(float* out, float* x, float *w, int n, int d);

// Root Mean Square Normalization
// Parameters:
//  out:    Output vector (size n)
//  x:      Input vector (size n)
//  weight: Scaling weights vector (size n)
//  n:      Number of elements
void RMSNorm(float* out, float* x, float* weight, int n);

// Rotary Positional Embedding
// Parameters:
//  q:      Query vector (size dim)
//  k:      Key vector (size kv_dim)
//  pos:    Position index
//  dim:    Query dimension
//  kv_dim: Key dimension
//  head_size: Head size
void rope(float* q, float* k, int pos, int dim, int kv_dim, int head_size, float rope_base);

// Softmax Function
// Parameters:
//  x:    Input vector (modified in-place to become probabilities)
//  size: Number of elements in the vector
void softmax(float* x, int size);

// SwiGLU Activation Function
// Calculates: hb = (h1 * sigmoid(h1)) * h3
// Parameters:
//  hb:   Output buffer (can alias h1 or h3)
//  h1:   Gate projection vector
//  h3:   Up projection vector
//  size: Dimension of vectors
void swiglu(float* hb, float* h1, float* h3, int size);
#endif