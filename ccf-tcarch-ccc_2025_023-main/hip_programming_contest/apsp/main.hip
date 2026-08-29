#include "main.h"

#define INF ((1 << 30) - 1)

#define BLOCK_DIM 16

#define HIP_CHECK(command) do { \
    hipError_t error = command; \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 1：更新主对角线上的块
// 只更新对角线上的 一个(k_block_idx, k_block_idx) 块
__global__ void floyd_warshall_block_kernel_phase1(int n, int k_block_idx, int* dist) {
    const unsigned int thread_row_in_block = threadIdx.y;
    const unsigned int thread_col_in_block = threadIdx.x;

    __shared__ int pivot_block[BLOCK_DIM][BLOCK_DIM];

    int block_start_row = k_block_idx * BLOCK_DIM;
    int block_start_col = k_block_idx * BLOCK_DIM;

    int global_row = block_start_row + thread_row_in_block;
    int global_col = block_start_col + thread_col_in_block;

    pivot_block[thread_row_in_block][thread_col_in_block] = dist[global_row * n + global_col];
    
    __syncthreads();

    // 在共享内存中对枢纽块进行内部的 Floyd-Warshall 更新
    for (int p = 0; p < BLOCK_DIM; ++p) {
        int path_sum = pivot_block[thread_row_in_block][p] + pivot_block[p][thread_col_in_block];
        if (pivot_block[thread_row_in_block][thread_col_in_block] > path_sum) {
            pivot_block[thread_row_in_block][thread_col_in_block] = path_sum;
        }
        __syncthreads();
    }

    dist[global_row * n + global_col] = pivot_block[thread_row_in_block][thread_col_in_block];
}


// 2：更新与枢纽块在同一行或同一列的块
__global__ void floyd_warshall_block_kernel_phase2(int n, int k_block_idx, int* dist) {
    const unsigned int thread_row_in_block = threadIdx.y;
    const unsigned int thread_col_in_block = threadIdx.x;

    const unsigned int i_block_idx = blockIdx.x; 

    // 如果当前块是枢纽块，则跳过
    if (i_block_idx == k_block_idx) return; 

    __shared__ int pivot_block_data[BLOCK_DIM][BLOCK_DIM];
    __shared__ int target_block_data[BLOCK_DIM][BLOCK_DIM];

    int pivot_global_row = k_block_idx * BLOCK_DIM + thread_row_in_block;
    int pivot_global_col = k_block_idx * BLOCK_DIM + thread_col_in_block;
    pivot_block_data[thread_row_in_block][thread_col_in_block] = dist[pivot_global_row * n + pivot_global_col];
    
    int target_global_row = k_block_idx * BLOCK_DIM + thread_row_in_block;
    int target_global_col = i_block_idx * BLOCK_DIM + thread_col_in_block;
    target_block_data[thread_row_in_block][thread_col_in_block] = dist[target_global_row * n + target_global_col];
    __syncthreads();

    // 更新 k 行上的块
    int val = target_block_data[thread_row_in_block][thread_col_in_block];
    for (int p = 0; p < BLOCK_DIM; p++) {
        int path_sum = pivot_block_data[thread_row_in_block][p] + target_block_data[p][thread_col_in_block];
        if (val > path_sum) {
            val = path_sum;
        }
    }
    dist[target_global_row * n + target_global_col] = val;

    __syncthreads();

  
    target_global_row = i_block_idx * BLOCK_DIM + thread_row_in_block;
    target_global_col = k_block_idx * BLOCK_DIM + thread_col_in_block;
    target_block_data[thread_row_in_block][thread_col_in_block] = dist[target_global_row * n + target_global_col];
   
    __syncthreads();

    // 更新 k 列上的块
    val = target_block_data[thread_row_in_block][thread_col_in_block];
    for (int p = 0; p < BLOCK_DIM; p++) {
        int path_sum = target_block_data[thread_row_in_block][p] + pivot_block_data[p][thread_col_in_block];
        if (val > path_sum) {
            val = path_sum;
        }
    }
  
    dist[target_global_row * n + target_global_col] = val;
}

// 3：更新所有剩余的非枢纽行和列的块
__global__ void floyd_warshall_block_kernel_phase3(int n, int k_block_idx, int* dist) {
   
    const unsigned int thread_row_in_block = threadIdx.y;
    const unsigned int thread_col_in_block = threadIdx.x;
  
    const unsigned int i_block_idx = blockIdx.y;
    const unsigned int j_block_idx = blockIdx.x;

    // 如果当前块在 k 行或 k 列上，则跳过，因为阶段2已经处理了
    if (i_block_idx == k_block_idx || j_block_idx == k_block_idx) return;

   
    __shared__ int block_A[BLOCK_DIM][BLOCK_DIM]; 
    __shared__ int block_B[BLOCK_DIM][BLOCK_DIM]; 

 
    int global_row_A = i_block_idx * BLOCK_DIM + thread_row_in_block;
    int global_col_A = k_block_idx * BLOCK_DIM + thread_col_in_block;
    block_A[thread_row_in_block][thread_col_in_block] = dist[global_row_A * n + global_col_A];

  
    int global_row_B = k_block_idx * BLOCK_DIM + thread_row_in_block;
    int global_col_B = j_block_idx * BLOCK_DIM + thread_col_in_block;
    block_B[thread_row_in_block][thread_col_in_block] = dist[global_row_B * n + global_col_B];

    __syncthreads();

   
    int global_row_C = i_block_idx * BLOCK_DIM + thread_row_in_block;
    int global_col_C = j_block_idx * BLOCK_DIM + thread_col_in_block;
    int current_dist = dist[global_row_C * n + global_col_C];

   
    for (int p = 0; p < BLOCK_DIM; ++p) {
        int path_sum = block_A[thread_row_in_block][p] + block_B[p][thread_col_in_block];
        if (current_dist > path_sum) {
            current_dist = path_sum;
        }
    }

    dist[global_row_C * n + global_col_C] = current_dist;
}



int main(int argc, char* argv[]) {
    // 1. 参数检查
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // 2. 文件读取
    std::ifstream infile(argv[1]);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open input file " << argv[1] << std::endl;
        return 1;
    }

    // 3. 读取图数据
    int V_orig, E;
    infile >> V_orig >> E;

    // 4. 矩阵填充
    int V_padded = ((V_orig + BLOCK_DIM - 1) / BLOCK_DIM) * BLOCK_DIM;

    // 5. 创建并初始化距离矩阵
    std::vector<int> h_dist(V_padded * V_padded);

    for (int i = 0; i < V_padded; ++i) {
        for (int j = 0; j < V_padded; ++j) {
            if (i < V_orig && j < V_orig) {
                h_dist[i * V_padded + j] = (i == j) ? 0 : INF;
            } else {
                h_dist[i * V_padded + j] = INF; 
            }
        }
    }

    // 读取边数据并更新距离矩阵
    for (int i = 0; i < E; ++i) {
        int u, v, w;
        infile >> u >> v >> w;
        if (u < V_orig && v < V_orig) {
            h_dist[u * V_padded + v] = w;
        }
    }
    infile.close();

    // 6. 设备内存分配和数据传输
    int* d_dist;
    size_t matrix_size = (size_t)V_padded * V_padded * sizeof(int);


    HIP_CHECK(hipMalloc(&d_dist, matrix_size));

    HIP_CHECK(hipMemcpy(d_dist, h_dist.data(), matrix_size, hipMemcpyHostToDevice));

    // 7. 设置启动参数和核函数调用
    const int num_blocks = V_padded / BLOCK_DIM;
  
    dim3 block_dims(BLOCK_DIM, BLOCK_DIM);
  
    dim3 grid_dims_phase2(num_blocks);
    dim3 grid_dims_phase3(num_blocks, num_blocks);
    
    // 8. 主循环：按块迭代
    for (int k_block = 0; k_block < num_blocks; ++k_block) {
        // 处理主对角线上的枢纽块
        hipLaunchKernelGGL(floyd_warshall_block_kernel_phase1, dim3(1), block_dims, 0, 0, V_padded, k_block, d_dist);
        HIP_CHECK(hipGetLastError());

        // 处理枢纽行和枢纽列上的其他块
        hipLaunchKernelGGL(floyd_warshall_block_kernel_phase2, grid_dims_phase2, block_dims, 0, 0, V_padded, k_block, d_dist);
        HIP_CHECK(hipGetLastError());

        // 处理所有剩余的块
        hipLaunchKernelGGL(floyd_warshall_block_kernel_phase3, grid_dims_phase3, block_dims, 0, 0, V_padded, k_block, d_dist);
        HIP_CHECK(hipGetLastError());
    }
    // 等待所有 GPU 任务完成
    HIP_CHECK(hipDeviceSynchronize());

    // 9. 结果回传
    HIP_CHECK(hipMemcpy(h_dist.data(), d_dist, matrix_size, hipMemcpyDeviceToHost));

    // 10. 结果输出
    for (int i = 0; i < V_orig; ++i) {
        for (int j = 0; j < V_orig; ++j) {
            std::cout << h_dist[i * V_padded + j] << (j == V_orig - 1 ? "" : " ");
        }
        std::cout << std::endl;
    }

    // 11. 释放设备内存
    HIP_CHECK(hipFree(d_dist));

    return 0;
}