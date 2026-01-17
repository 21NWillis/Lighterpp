#include "kernels.cuh"

#define BLOCK_SIZE 256


__global__ void scale_kernel_multihead(float* att_scores, float scale, int n_heads, int seq_len, int att_stride) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;     
    int head = blockIdx.y; 
    
    if (i >= seq_len || head >= n_heads) return;
    
    float* att_head = att_scores + head * att_stride;
    att_head[i] *= scale;
}

void cuda_scale_multihead(float* d_att, float scale, int n_heads, int seq_len, int att_stride) {
    dim3 grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads);
    dim3 block(BLOCK_SIZE);
    
    scale_kernel_multihead<<<grid, block>>>(d_att, scale, n_heads, seq_len, att_stride);
}
