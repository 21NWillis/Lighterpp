#ifndef KERNELS_CUH
#define KERNELS_CUH

// CUDA GEMV (General Matrix-Vector multiply)
// Computes: out = W * x
// Parameters:
//   out: Output vector (size n)
//   x:   Input vector (size d)  
//   w:   Weight matrix (n rows, d columns) - flattened, row-major
//   n:   Number of rows (output dimension)
//   d:   Number of columns (input dimension)
void cuda_gemv(float* out, float* x, float* w, int n, int d);

#endif
