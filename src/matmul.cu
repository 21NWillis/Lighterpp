#include "kernels.cuh"
#include <cuda_runtime.h>

// Naive CUDA GEMV Kernel
// Parameters:
//   out: Output vector (size n)
//   x:   Input vector (size d)  
//   w:   Weight matrix (n rows, d columns) - flattened, row-major
//   n:   Number of rows (output dimension)
//   d:   Number of columns (input dimension)
__global__ void gemv_kernel_naive(float* out, float* x, float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float sum = 0.0f;
        for (int k = 0; k < d; k++) {
            sum += x[k] * w[i * d + k];
        }
        out[i] = sum;
    }
}

// Host wrapper function
// Parameters:
//   out: Output vector (size n)
//   x:   Input vector (size d)  
//   w:   Weight matrix (n rows, d columns) - flattened, row-major
//   n:   Number of rows (output dimension)
//   d:   Number of columns (input dimension)
void cuda_gemv(float* out, float* x, float* w, int n, int d) {
    float *d_out, *d_x, *d_w;
    
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMalloc(&d_x, d * sizeof(float));
    cudaMalloc(&d_w, n * d * sizeof(float));
    
    cudaMemcpy(d_x, x, d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, n * d * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    gemv_kernel_naive<<<numBlocks, blockSize>>>(d_out, d_x, d_w, n, d);
    
    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_out);
    cudaFree(d_x);
    cudaFree(d_w);
}
