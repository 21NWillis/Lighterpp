#include "kernels.cuh"
#include <cuda_fp16.h>

#define HEAD_SIZE_MAX 128

__global__ void multihead_gemv_kernel_optimized(float* d_out, float* d_q, __half* d_k_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_size, int seq_len) {
    __shared__ __align__(16) float q_shared[HEAD_SIZE_MAX];
    
    int h = blockIdx.y;
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    
    int tid_linear = threadIdx.y * 32 + threadIdx.x;
    if (tid_linear < head_size) {
        q_shared[tid_linear] = d_q[h * head_size + tid_linear];
    }
    
    __syncthreads();
    
    int t = blockIdx.x * blockDim.y + warp_id;
    
    if (t > pos) return;

    float* q_head = q_shared;
    
    int kv_head = h / (n_heads / n_kv_heads);
    long long layer_offset = (long long)layer * n_kv_heads * seq_len * head_size;
    long long head_offset = (long long)kv_head * seq_len * head_size;
    long long token_offset = (long long)t * head_size;
    __half* k_vec = d_k_cache + layer_offset + head_offset + token_offset;
    
    float sum = 0.0f;
    int num_pairs = head_size / 2;
    
    for (int i = lane_id; i < num_pairs; i += 32) {
        half2 k_h2 = reinterpret_cast<half2*>(k_vec)[i];
        float2 k_f2 = __half22float2(k_h2);
        
        float q0 = q_head[i * 2];
        float q1 = q_head[i * 2 + 1];
        
        sum += q0 * k_f2.x + q1 * k_f2.y;
    }
    
    if (head_size % 2 != 0 && lane_id == 0) {
        int last = head_size - 1;
        sum += q_head[last] * __half2float(k_vec[last]);
    }
    
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    
    if (lane_id == 0) {
        d_out[h * seq_len + t] = sum;
    }
}

void cuda_multihead_gemv(float* d_out, float* d_q, __half* d_k_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_size, int seq_len) {
    dim3 blockDim(32, 4);
    int warps_per_block = 4;
    int num_blocks_t = (pos + 1 + warps_per_block - 1) / warps_per_block;
    dim3 gridDim(num_blocks_t, n_heads);

    multihead_gemv_kernel_optimized<<<gridDim, blockDim>>>(d_out, d_q, d_k_cache, layer, pos, n_heads, n_kv_heads, head_size, seq_len);
}
