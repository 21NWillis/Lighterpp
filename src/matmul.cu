#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  
#include "model.h"  // For WeightPrecision enum
#include <stdint.h>

#define BLOCK_SIZE 256

__global__ void gemv_kernel(float* out, float* x, float* w, int n, int d) {
    __shared__ float warp_sums[32];
    // Use defined limit (40KB=10240 floats)
    __shared__ float x_shared[MAX_SHARED_FLOATS];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;

    for (int i = tid; i < d; i += blockDim.x) {
        x_shared[i] = x[i];
    }

    __syncthreads();
    
    float* w_row = w + row * d;
    
    float local_sum = 0.0f;
    int d4 = d / 4;
    
    for (int i = tid; i < d4; i += blockDim.x) {
        float4 w_vec = reinterpret_cast<float4*>(w_row)[i];
        float4 x_vec = reinterpret_cast<float4*>(x_shared)[i];
        
        local_sum += w_vec.x * x_vec.x;
        local_sum += w_vec.y * x_vec.y;
        local_sum += w_vec.z * x_vec.z;
        local_sum += w_vec.w * x_vec.w;
    }
    int base = d4 * 4;
    for (int i = base + tid; i < d; i += blockDim.x) {
        local_sum += w_row[i] * x_shared[i];
    }

    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 16);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 8);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 4);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 2);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 1);
    
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_total = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 16);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 8);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 4);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 2);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 1);
        
        if (lane_id == 0) {
            out[row] = warp_total;
        }
    }
}



__global__ void gemv_f16_kernel(float* out, __half* x, __half* w, int n, int d) {
    __shared__ float warp_sums[32];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;
    
    __half* w_row = w + row * d;
    float local_sum = 0.0f;
    
    // Initialize FP16 accumulator
    half2 sum_h2 = __float2half2_rn(0.0f);
    
    int vec_loops = d / 8;
    for (int i = tid; i < vec_loops; i += blockDim.x) {
        float4 w_vec_raw = reinterpret_cast<float4*>(w_row)[i];
        float4 x_vec_raw = reinterpret_cast<float4*>(x)[i];
        
        half2* w_h2 = reinterpret_cast<half2*>(&w_vec_raw);
        half2* x_h2 = reinterpret_cast<half2*>(&x_vec_raw);
        
        sum_h2 = __hfma2(w_h2[0], x_h2[0], sum_h2);
        sum_h2 = __hfma2(w_h2[1], x_h2[1], sum_h2);
        sum_h2 = __hfma2(w_h2[2], x_h2[2], sum_h2);
        sum_h2 = __hfma2(w_h2[3], x_h2[3], sum_h2);
    }
    
    float2 sum_f2 = __half22float2(sum_h2);
    local_sum += sum_f2.x + sum_f2.y;
    
    int start_rem = vec_loops * 8;
    for (int i = start_rem + tid; i < d; i += blockDim.x) {
        local_sum += __half2float(w_row[i]) * __half2float(x[i]);
    }
    
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 16);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 8);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 4);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 2);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 1);
    
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_total = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 16);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 8);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 4);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 2);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 1);
        
        if (lane_id == 0) {
            out[row] = warp_total;
        }
    }
}

void cuda_gemv(float* d_out, float* d_x, void* d_w, void* d_workspace, int n, int d, WeightPrecision precision) {
    switch (precision) {
        case WEIGHT_FP16: {
            cuda_convert_f32_to_f16(d_workspace, d_x, d);
            gemv_f16_kernel<<<n, BLOCK_SIZE>>>(d_out, (__half*)d_workspace, (__half*)d_w, n, d);
            checkCuda(cudaGetLastError());
            break;
        }
        case WEIGHT_FP32:
        default:
            gemv_kernel<<<n, BLOCK_SIZE>>>(d_out, d_x, (float*)d_w, n, d);
            break;
    }
}

// Helper kernel for test setup (F32 -> F16)
__global__ void convert_f32_to_f16_kernel(__half* dst, float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

void cuda_convert_f32_to_f16(void* dst, float* src, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    convert_f32_to_f16_kernel<<<blocks, threads>>>((__half*)dst, src, n);
}
