#include "kernels.cuh"

#define BLOCK_SIZE 256

__global__ void residual_add_kernel(float* out, const float* a, const float* b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int n4 = n / 4;
    if (tid < n4) {
        float4 a_vec = reinterpret_cast<const float4*>(a)[tid];
        float4 b_vec = reinterpret_cast<const float4*>(b)[tid];
        float4 out_vec;
        out_vec.x = a_vec.x + b_vec.x;
        out_vec.y = a_vec.y + b_vec.y;
        out_vec.z = a_vec.z + b_vec.z;
        out_vec.w = a_vec.w + b_vec.w;
        reinterpret_cast<float4*>(out)[tid] = out_vec;
    }
    
    int remainder_idx = n4 * 4 + tid;
    if (remainder_idx < n && tid < (n - n4 * 4)) {
        out[remainder_idx] = a[remainder_idx] + b[remainder_idx];
    }
}

void cuda_residual_add(float* d_out, const float* d_a, const float* d_b, int n) {
    int n4 = n / 4;
    int num_blocks = (n4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks == 0) num_blocks = 1;
    residual_add_kernel<<<num_blocks, BLOCK_SIZE>>>(d_out, d_a, d_b, n);
}
