#include "main.h"

using DataType = long long;

#define HIP_CHECK(call) do { \
  hipError_t err = call; \
  if (err != hipSuccess) { \
    fprintf(stderr,"HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

constexpr int BLOCK_SIZE = 512;               
constexpr int WARP_SIZE = 64;                 
constexpr int ELEMENTS_PER_THREAD = 1;       
constexpr int ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD;


// Warp 内 Inclusive Scan (基于 Shuffle)
__device__ __forceinline__
DataType warp_inclusive_scan(DataType val) {
    // __shfl_up 在 HIP 中用于获取同一 warp 中前面线程的寄存器值
    // 循环展开的方式进行并行前缀和
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        DataType y = __shfl_up(val, offset, WARP_SIZE);
        // 只有当线程的 lane id 大于等于 offset 时，才加上前面线程传来的值
        if ((int(threadIdx.x) % WARP_SIZE) >= offset) {
            val += y;
        }
    }
    return val;
}


// 块内前缀和 (int 输入版本)
__global__ void scan_block_int_input(const int* in, DataType* out, DataType* block_sums, int N) {
    // 共享内存，于存储每个 warp 的扫描总和
    __shared__ DataType warp_sums[BLOCK_SIZE / WARP_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    DataType val = (gid < N) ? static_cast<DataType>(in[gid]) : 0;

    // Warp 内扫描
    val = warp_inclusive_scan(val);

    // 各 warp 的最后一个线程将 warp 总和存入共享内存
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == WARP_SIZE - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads(); // 确保所有 warp 的总和都已写入

    // Warp 0 扫描所有 warp_sums (串行前缀和)
    // 只有第一个 warp (Warp 0) 的线程参与计算
    if (warp_id == 0) {
        // Warp 0 的每个线程加载一个 warp_sum
        DataType warp_val = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane_id] : 0;
        warp_val = warp_inclusive_scan(warp_val);
        // 计算结果写回共享内存
        if (lane_id < (BLOCK_SIZE / WARP_SIZE)) {
            warp_sums[lane_id] = warp_val;
        }
    }
    __syncthreads();

    // 每个 warp 加上前面所有 warp 的总和偏移量
    if (warp_id > 0) {
        val += warp_sums[warp_id - 1];
    }

    // 将 inclusive scan 结果转换为 exclusive scan 结果
    if (gid < N) {
        out[gid] = val - static_cast<DataType>(in[gid]);
    }

    // 保存整个 block 的总和 (块内最后一个线程完成)
    if (block_sums && threadIdx.x == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = val;
    }
}


// 块内前缀和 (long long 输入版本, 递归时用)
__global__ void scan_block_ll_input(const DataType* in, DataType* out, DataType* block_sums, int N) {
    __shared__ DataType warp_sums[BLOCK_SIZE / WARP_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    DataType val = (gid < N) ? in[gid] : 0;

    val = warp_inclusive_scan(val);

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == WARP_SIZE - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        DataType warp_val = (lane_id < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane_id] : 0;
        warp_val = warp_inclusive_scan(warp_val);
        if (lane_id < (BLOCK_SIZE / WARP_SIZE)) {
            warp_sums[lane_id] = warp_val;
        }
    }
    __syncthreads();

    if (warp_id > 0) {
        val += warp_sums[warp_id - 1];
    }

    if (gid < N) {
        out[gid] = val - in[gid];
    }

    if (block_sums && threadIdx.x == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = val;
    }
}

// 偏移加法
__global__ void add_block_offsets(DataType* data, const DataType* block_offsets, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        data[gid] += block_offsets[blockIdx.x];
    }
}

// Inclusive + Cast 输出
__global__ void inclusive_cast_output(const DataType* exclusive, const int* input, int* output, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        DataType val = exclusive[gid] + static_cast<DataType>(input[gid]);
        output[gid] = static_cast<int>(val);
    }
}

void scan_recursive_ll(const DataType* d_input, DataType* d_exclusive, int N, DataType* workspace);

void scan_recursive_int(const int* d_input, DataType* d_exclusive, int N, DataType* workspace) {
    int num_blocks = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    if (num_blocks == 1) {
        hipLaunchKernelGGL(scan_block_int_input, dim3(1), dim3(BLOCK_SIZE), 0, 0,
                           d_input, d_exclusive, nullptr, N);
        HIP_CHECK(hipGetLastError());
    } else {
        DataType* d_block_sums = workspace;
        DataType* d_scanned_sums = workspace + num_blocks;

        hipLaunchKernelGGL(scan_block_int_input, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                           d_input, d_exclusive, d_block_sums, N);
        HIP_CHECK(hipGetLastError());

        scan_recursive_ll(d_block_sums, d_scanned_sums, num_blocks, workspace + 2 * num_blocks);

        hipLaunchKernelGGL(add_block_offsets, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                           d_exclusive, d_scanned_sums, N);
        HIP_CHECK(hipGetLastError());
    }
}

void scan_recursive_ll(const DataType* d_input, DataType* d_exclusive, int N, DataType* workspace) {
    int num_blocks = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    if (num_blocks == 1) {
        hipLaunchKernelGGL(scan_block_ll_input, dim3(1), dim3(BLOCK_SIZE), 0, 0,
                           d_input, d_exclusive, nullptr, N);
        HIP_CHECK(hipGetLastError());
    } else {
        DataType* d_block_sums = workspace;
        DataType* d_scanned_sums = workspace + num_blocks;
        
        hipLaunchKernelGGL(scan_block_ll_input, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                           d_input, d_exclusive, d_block_sums, N);
        HIP_CHECK(hipGetLastError());

        scan_recursive_ll(d_block_sums, d_scanned_sums, num_blocks, workspace + 2 * num_blocks);

        hipLaunchKernelGGL(add_block_offsets, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                           d_exclusive, d_scanned_sums, N);
        HIP_CHECK(hipGetLastError());
    }
}

// solve 函数 
extern "C" void solve(const int* input, int* output, int N) {
    if (N <= 0) return;

    size_t size_int = (size_t)N * sizeof(int);
    size_t size_ll = (size_t)N * sizeof(DataType);

    int *d_input, *d_output;
    DataType *d_exclusive, *d_workspace = nullptr;
    HIP_CHECK(hipMalloc(&d_input, size_int));
    HIP_CHECK(hipMalloc(&d_output, size_int));
    HIP_CHECK(hipMalloc(&d_exclusive, size_ll));

    // 计算并分配工作空间 
    size_t workspace_size = 0;
    int current_n = N;
    while (current_n > ELEMENTS_PER_BLOCK) {
        int num_blocks = (current_n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        workspace_size += 2 * num_blocks;
        current_n = num_blocks;
    }

    if (workspace_size > 0) {
        HIP_CHECK(hipMalloc(&d_workspace, workspace_size * sizeof(DataType)));
    }

    HIP_CHECK(hipMemcpy(d_input, input, size_int, hipMemcpyHostToDevice));

    // 启动递归扫描
    scan_recursive_int(d_input, d_exclusive, N, d_workspace);

    // 启动最终的融合内核
    int grid_size = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    hipLaunchKernelGGL(inclusive_cast_output, dim3(grid_size), dim3(BLOCK_SIZE), 0, 0,
                       d_exclusive, d_input, d_output, N);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(output, d_output, size_int, hipMemcpyDeviceToHost));

    // 释放所有内存
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_exclusive));
    if (d_workspace != nullptr) {
        HIP_CHECK(hipFree(d_workspace));
    }
}