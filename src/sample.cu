
#include "kernels.cuh"

#define BLOCK_SIZE 256

__global__ void fused_sample_kernel(float* logits, int vocab_size, float temperature, float penalty, const int* history, int history_len, float rand_val, int* result) {
    __shared__ float warp_maxes[8];
    __shared__ int warp_indices[8];
    __shared__ float warp_sums[8];
    __shared__ float global_max;
    __shared__ float global_sum;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    
    if (penalty > 1.0f && history != nullptr) {
        for (int i = tid; i < history_len; i += blockDim.x) {
            int token = history[i];
            if (token >= 0 && token < vocab_size) {
                float val = logits[token];
                logits[token] = (val > 0) ? val / penalty : val * penalty;
            }
        }
        __syncthreads();
    }
    
    float inv_temp = (temperature > 0) ? (1.0f / temperature) : 1.0f;
    float local_max = -INFINITY;
    int local_max_idx = 0;
    
    int stride = blockDim.x * 4;
    
    for (int i = tid*4; i < vocab_size; i += stride) {
        float4 vec = reinterpret_cast<float4*>(logits)[i/4];
        vec.x *= inv_temp;
        vec.y *= inv_temp;
        vec.z *= inv_temp;
        vec.w *= inv_temp;

        reinterpret_cast<float4*>(logits)[i/4] = vec;
        if (vec.x > local_max) {
            local_max = vec.x;
            local_max_idx = i;
        }
        if (vec.y > local_max) {
            local_max = vec.y;
            local_max_idx = i+1;
        }
        if (vec.z > local_max) {
            local_max = vec.z;
            local_max_idx = i+2;
        }
        if (vec.w > local_max) {
            local_max = vec.w;
            local_max_idx = i+3;
        }
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, local_max_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_max_idx = other_idx;
        }
    }

    if (lane_id == 0) {
        warp_maxes[warp_id] = local_max;
        warp_indices[warp_id] = local_max_idx;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_max = (lane_id < num_warps) ? warp_maxes[lane_id] : -INFINITY;
        int warp_idx = (lane_id < num_warps) ? warp_indices[lane_id] : 0;
        
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 16));
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 8));
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 4));
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 2));
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 1));
        
        if (lane_id == 0) {
            global_max = warp_max;
            warp_indices[0] = warp_idx; 
        }
    }
    __syncthreads();
    
    if (temperature == 0.0f) {
        if (tid == 0) {
            result[0] = warp_indices[0]; 
        }
        return;
    }
    
    float local_sum = 0.0f;
    
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float e = __expf(logits[i] - global_max);
        logits[i] = e;
        local_sum += e;
    }

    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 16);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 8);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 4);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 2);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 1);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 16);
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 8);
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 4);
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 2);
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 1);
        
        if (lane_id == 0) {
            global_sum = warp_sum;
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        float threshold = rand_val * global_sum;
        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += logits[i];
            if (cumsum > threshold) {
                result[0] = i;
                return;
            }
        }
        result[0] = vocab_size - 1; 
    }
}

void cuda_sample(float* d_logits, int vocab_size, float temperature, float topp, float penalty, int* d_history, int history_len, int* d_sampled_token, unsigned int* d_rng_state) {
    float rand_val = (float)rand() / (float)RAND_MAX;
    
    fused_sample_kernel<<<1, BLOCK_SIZE>>>(d_logits, vocab_size, temperature, penalty, d_history, history_len, rand_val, d_sampled_token);
}
