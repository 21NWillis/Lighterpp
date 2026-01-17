#include "kernels.cuh"

#define BLOCK_SIZE 256

__global__ void softmax_multihead(float* out, float* x, int n_heads, int seq_len, int att_stride) {
    int head = blockIdx.y;  
    float* x_head = x + head * att_stride;         
    float* out_head = out + head * att_stride;     
    int n = seq_len;           
    
    __shared__ float warp_vals[32];
    __shared__ float global_max;
    __shared__ float global_sum;
    __shared__ float inv_sum;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int n4 = n / 4;

    // Find max value
    float local_max = -INFINITY;
    #pragma unroll
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 x_vec = reinterpret_cast<float4*>(x_head)[i];
        local_max = fmaxf(local_max, x_vec.x);
        local_max = fmaxf(local_max, x_vec.y);
        local_max = fmaxf(local_max, x_vec.z);
        local_max = fmaxf(local_max, x_vec.w);
    }
    for (int i = n4*4 + tid; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, x_head[i]);
    }

    // Warp reduction for max
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, 16));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, 8));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, 4));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, 2));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, 1));

    if (lane_id == 0) warp_vals[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        float warp_max = (lane_id < num_warps) ? warp_vals[lane_id] : -INFINITY;
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 16));
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 8));
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 4));
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 2));
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFF, warp_max, 1));
        if (lane_id == 0) global_max = warp_max;
    }
    __syncthreads();

    // Compute exp(x - max) and accumulate sum
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 x_vec = reinterpret_cast<float4*>(x_head)[i];
        float4 exp_vec;
        exp_vec.x = __expf(x_vec.x - global_max);
        exp_vec.y = __expf(x_vec.y - global_max);
        exp_vec.z = __expf(x_vec.z - global_max);
        exp_vec.w = __expf(x_vec.w - global_max);
        reinterpret_cast<float4*>(out_head)[i] = exp_vec;
        local_sum += exp_vec.x + exp_vec.y + exp_vec.z + exp_vec.w;
    }
    for (int i = n4*4 + tid; i < n; i += blockDim.x) {
        float val = __expf(x_head[i] - global_max);
        out_head[i] = val;
        local_sum += val;
    }

    // Warp reduction for sum
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 16);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 8);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 4);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 2);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 1);

    if (lane_id == 0) warp_vals[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        float warp_sum = (lane_id < num_warps) ? warp_vals[lane_id] : 0.0f;
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 16);
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 8);
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 4);
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 2);
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 1);
        if (lane_id == 0) global_sum = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        inv_sum = __fdividef(1.0f, global_sum);
    }
    __syncthreads();

    // Final division by sum
    #pragma unroll
    for (int i = tid; i < n4; i += blockDim.x) {
        float4 out_vec = reinterpret_cast<float4*>(out_head)[i];
        float4 div_vec;
        div_vec.x = out_vec.x * inv_sum;
        div_vec.y = out_vec.y * inv_sum;
        div_vec.z = out_vec.z * inv_sum;
        div_vec.w = out_vec.w * inv_sum;
        reinterpret_cast<float4*>(out_head)[i] = div_vec;
    }
    for (int i = n4*4 + tid; i < n; i += blockDim.x) {
        out_head[i] = out_head[i] * inv_sum;
    }
}

// Host wrapper for multi-head softmax
void cuda_softmax_multihead(float* d_out, float* d_x, int n_heads, int seq_len, int att_stride) {
    
    dim3 grid(1, n_heads);  
    dim3 block(BLOCK_SIZE);
    
    softmax_multihead<<<grid, block>>>(d_out, d_x, n_heads, seq_len, att_stride);
}
