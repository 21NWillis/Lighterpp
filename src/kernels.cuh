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

#endif
