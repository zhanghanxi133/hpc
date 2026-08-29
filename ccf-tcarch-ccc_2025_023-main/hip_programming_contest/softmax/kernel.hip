#include "main.h"
#include <math.h>

#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float sdata[1024]; // 假设最大块大小为1024
    unsigned int tx = threadIdx.x;
    sdata[tx] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
        __syncthreads();
    }
    return sdata[0];
}

__device__ __forceinline__ float blockReduceMax(float val) {
    __shared__ float sdata[1024]; // 假设最大块大小为1024
    unsigned int tx = threadIdx.x;
    sdata[tx] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            sdata[tx] = fmaxf(sdata[tx], sdata[tx + s]);
        }
        __syncthreads();
    }
    return sdata[0];
}

// 计算局部最大值和归一化和
__global__ void partial_softmax_kernel(const float* __restrict__ input, 
                                       float* __restrict__ partial_maxs,
                                       float* __restrict__ partial_norms,
                                       int N) {
    unsigned int tid = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int grid_size = gridDim.x;

    // 每个线程处理的元素索引范围
    unsigned int global_tid = block_id * block_size + tid;
    
    // 局部归约求最大值
    float local_max = -FLT_MAX;
    for (unsigned int i = global_tid; i < N; i += block_size * grid_size) {
        local_max = fmaxf(local_max, input[i]);
    }
    
    float block_max = blockReduceMax(local_max);
    
    // 局部归约求归一化和
    float local_norm = 0.0f;
    if (block_max != -FLT_MAX) { // 避免空块的计算
        for (unsigned int i = global_tid; i < N; i += block_size * grid_size) {
            local_norm += expf(input[i] - block_max);
        }
    }

    float block_norm = blockReduceSum(local_norm);
    
    // 只有块内的第一个线程写入结果
    if (tid == 0) {
        partial_maxs[block_id] = block_max;
        partial_norms[block_id] = block_norm;
    }
}

// 全局归约和最终Softmax计算
__global__ void final_softmax_kernel(const float* __restrict__ input, 
                                     const float* __restrict__ partial_maxs,
                                     const float* __restrict__ partial_norms,
                                     float* __restrict__ output,
                                     int N, 
                                     int num_blocks) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    if (blockIdx.x > 0) return;

    float local_max = (tid < num_blocks) ? partial_maxs[tid] : -FLT_MAX;
    float local_norm = (tid < num_blocks) ? partial_norms[tid] : 0.0f;

    float global_max = blockReduceMax(local_max);
    
    // 全局归一化和的计算，考虑最大值变化
    float global_norm = 0.0f;
    if (global_max != -FLT_MAX) {
        float thread_norm = 0.0f;
        for (unsigned int i = tid; i < num_blocks; i += block_size) {
            thread_norm += partial_norms[i] * expf(partial_maxs[i] - global_max);
        }
        global_norm = blockReduceSum(thread_norm);
    }
    
    __syncthreads();

    for (unsigned int i = tid; i < N; i += block_size) {
        output[i] = expf(input[i] - global_max) / global_norm;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    
    float *d_input, *d_output, *d_partial_maxs, *d_partial_norms;
    
    int threads_per_block = 256;
    int num_blocks = CEIL_DIV(N, threads_per_block); 

    HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_partial_maxs, num_blocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_partial_norms, num_blocks * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(partial_softmax_kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                      d_input, d_partial_maxs, d_partial_norms, N);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(final_softmax_kernel, dim3(1), dim3(threads_per_block), 0, 0,
                      d_input, d_partial_maxs, d_partial_norms, d_output, N, num_blocks);
    HIP_CHECK(hipGetLastError());
    
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_partial_maxs));
    HIP_CHECK(hipFree(d_partial_norms));
}

// FINAL RESULT: ALL TESTS PASSED!

//     Passed cases: 10 / 10
//     Total execution time: 5.92 seconds

// Job finished at Wed Sep 10 10:38:56 AM UTC 2025