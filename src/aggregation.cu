#include "kernels.cuh"

#define BLOCK_SIZE 256

__global__ void aggregation_kernel_multihead(float* out, const float* value_cache, const float* att_scores, int n_heads, int seq_len, int head_size, int gqa_factor, int att_stride) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int head = blockIdx.y;
    
    if (i >= head_size || head >= n_heads) return;
    
    int kv_head = head / gqa_factor;
    
    const float* att_head = att_scores + head * att_stride;
    const float* v_cache_head = value_cache + kv_head * (att_stride * head_size);
    float* out_head = out + head * head_size;
    
    float sum = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        sum += v_cache_head[t * head_size + i] * att_head[t];
    }
    
    out_head[i] = sum;
}

void cuda_aggregation_multihead(float* d_out, const float* d_v, const float* d_att, int n_heads, int seq_len, int head_size, int gqa_factor, int att_stride) {
    dim3 grid((head_size + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads);
    dim3 block(BLOCK_SIZE);
    
    aggregation_kernel_multihead<<<grid, block>>>(d_out, d_v, d_att, n_heads, seq_len, head_size, gqa_factor, att_stride);
}

