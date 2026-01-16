#include "kernels.cuh"

#define BLOCK_SIZE 256

__global__ void scatter_kv_kernel(float* key_cache, float* value_cache, const float* k, const float* v, int layer, int pos, int n_kv_heads, int head_size, int seq_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int kv_dim = n_kv_heads * head_size;
    
    if (idx >= kv_dim) return;
    
    int head = idx / head_size;
    int offset_in_head = idx % head_size;
    
    int cache_stride = seq_len * head_size;
    int dst_idx = layer * (kv_dim * seq_len) + head * cache_stride + pos * head_size + offset_in_head;
    
    key_cache[dst_idx] = k[idx];
    value_cache[dst_idx] = v[idx];
}

void cuda_scatter_kv(float* d_key_cache, float* d_value_cache, const float* d_k, const float* d_v, int layer, int pos, int n_kv_heads, int head_size, int seq_len) {
    int kv_dim = n_kv_heads * head_size;
    int num_blocks = (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scatter_kv_kernel<<<num_blocks, BLOCK_SIZE>>>(d_key_cache, d_value_cache, d_k, d_v, layer, pos, n_kv_heads, head_size, seq_len);
}
