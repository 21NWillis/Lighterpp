#include "kernels.cuh"

#define BLOCK_SIZE 256


__global__ void scale_kernel(float* x, float scale, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    

    int n4 = n / 4;
    if (tid < n4) {
        float4 vec = reinterpret_cast<float4*>(x)[tid];
        vec.x *= scale;
        vec.y *= scale;
        vec.z *= scale;
        vec.w *= scale;
        reinterpret_cast<float4*>(x)[tid] = vec;
    }
    
    int remainder_idx = n4 * 4 + tid;
    if (remainder_idx < n && tid < (n - n4 * 4)) {
        x[remainder_idx] *= scale;
    }
}

void cuda_scale(float* d_x, float scale, int n) {
    int n4 = n / 4;
    int num_blocks = (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks == 0) num_blocks = 1; 
    scale_kernel<<<num_blocks, BLOCK_SIZE>>>(d_x, scale, n);
}
