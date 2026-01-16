#include "kernels.cuh"

#define BLOCK_SIZE 256



__global__ void rope_kernel(float* q, float* k, int pos, int dim, int kv_dim, int head_size, float rope_base) {
    int pair_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (pair_idx >= dim / 2) return;

    int head_dim = (pair_idx * 2) % head_size;
    
    float freq = 1.0f / powf(rope_base, head_dim/float(head_size));
    float theta = pos * freq;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    float2 q_pair = reinterpret_cast<float2*>(q)[pair_idx];
    float2 q_rot;
    q_rot.x = q_pair.x * cos_theta - q_pair.y * sin_theta;
    q_rot.y = q_pair.x * sin_theta + q_pair.y * cos_theta;
    reinterpret_cast<float2*>(q)[pair_idx] = q_rot;

    if (pair_idx < kv_dim/2) {
        float2 k_pair = reinterpret_cast<float2*>(k)[pair_idx];
        float2 k_rot;
        k_rot.x = k_pair.x * cos_theta - k_pair.y * sin_theta;
        k_rot.y = k_pair.x * sin_theta + k_pair.y * cos_theta;
        reinterpret_cast<float2*>(k)[pair_idx] = k_rot;
    }
}

// Host wrapper function
void cuda_rope(float* d_q, float* d_k, int pos, int dim, int kv_dim, int head_size, float rope_base) {
    int num_pairs = dim / 2;
    int num_blocks = (num_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rope_kernel<<<num_blocks, BLOCK_SIZE>>>(d_q, d_k, pos, dim, kv_dim, head_size, rope_base);
}

