#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

enum WeightPrecision : int;

// Define maximum shared memory floats for kernels to prevent overflow
// 10240 floats = 40KB, safe for most GPUs (limit usually 48KB)
// Supports hidden_dim up to 10240
#define MAX_SHARED_FLOATS 10240

// Helper for CUDA error checking
#define checkCuda(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Kernel declarations matching .cu implementations
void cuda_convert_f32_to_f16(void* dst, float* src, int n);

void cuda_gemv(float* d_out, float* d_x, void* d_w, void* d_workspace, int n, int d, WeightPrecision precision);
void cuda_rmsnorm(float* d_out, float* d_x, float* d_w, int n);
void cuda_rope(float* d_q, float* d_k, int pos, int dim, int kv_dim, int head_size, float rope_base);
void cuda_swiglu(float* d_hb, float* d_h1, float* d_h3, int size);
void cuda_softmax_multihead(float* d_out, float* d_x, int n_heads, int seq_len, int att_stride);
void cuda_scale_multihead(float* d_att, float scale, int n_heads, int seq_len, int att_stride);
void cuda_aggregation_multihead(float* d_out, const float* d_v, const float* d_att, int n_heads, int seq_len, int head_size, int gqa_factor, int att_stride);
void cuda_residual_add(float* d_out, const float* d_a, const float* d_b, int n);
void cuda_scatter_kv(float* d_key_cache, float* d_value_cache, const float* d_k, const float* d_v, int layer, int pos, int n_kv_heads, int head_size, int seq_len);
void cuda_multihead_gemv(float* d_out, float* d_q, float* d_k_cache, int layer, int pos, int n_heads, int n_kv_heads, int head_size, int seq_len);
void cuda_sample(float* d_logits, int vocab_size,float temperature,float topp,float penalty,int* d_history,int history_len,int* d_sampled_token,unsigned int* d_rng_state);
