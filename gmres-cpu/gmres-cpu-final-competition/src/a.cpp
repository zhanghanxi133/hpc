#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cfloat>
#include <cstring>
#include "sparseMatrix.hpp"
#include "gmres.hpp"
#include <hip/hip_runtime.h>

using namespace std;

const int RESTART_TIMES = 20;
const double REL_RESID_LIMIT = 1e-6;
const int ITERATION_LIMIT = 10000;

// ============ 重命名常量 ============
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BASIS_VECTORS = RESTART_TIMES + 1;

#if defined(__HIP_PLATFORM_AMD__)
constexpr int WAVEFRONT_SIZE = 64;
#else
constexpr int WAVEFRONT_SIZE = 32;
#endif

#define CHECK_HIP_ERROR(cmd) \
do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %d\n", hipGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

inline int compute_grid_size(int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }
inline int compute_reduction_grid(int n) { 
    return std::max(1, std::min((n + 8*BLOCK_SIZE - 1)/(8*BLOCK_SIZE), 4096)); 
}

// ============ CPU端函数 - 重写版 ============

void applyGivensRotation(double &h1, double &h2, double c, double s) {
    double tmp = c * h1 + s * h2;
    h2 = -s * h1 + c * h2;
    h1 = tmp;
}

void computeGivensRotation(double &diagonal, double &subdiag, double &cosine, double &sine) {
    if (diagonal == 0.0) {
        cosine = 0.0;
        sine = 1.0;
    } else {
        double r = std::hypot(diagonal, subdiag);
        cosine = diagonal / r;
        sine = subdiag / r;
    }
}

void performGivensSequence(uint col_size, double *hessenberg, double *cos_vals, 
                          double *sin_vals, double *rhs_vec, uint col_idx) {
    for (uint row = 0; row < col_idx; row++) {
        double h_current = hessenberg[row * col_size + col_idx];
        double h_next = hessenberg[(row + 1) * col_size + col_idx];
        applyGivensRotation(h_current, h_next, cos_vals[row], sin_vals[row]);
        hessenberg[row * col_size + col_idx] = h_current;
        hessenberg[(row + 1) * col_size + col_idx] = h_next;
    }
    
    double h_diag = hessenberg[col_idx * col_size + col_idx];
    double h_sub = hessenberg[(col_idx + 1) * col_size + col_idx];
    computeGivensRotation(h_diag, h_sub, cos_vals[col_idx], sin_vals[col_idx]);
    
    applyGivensRotation(hessenberg[col_idx * col_size + col_idx], 
                       hessenberg[(col_idx + 1) * col_size + col_idx], 
                       cos_vals[col_idx], sin_vals[col_idx]);
    applyGivensRotation(rhs_vec[col_idx], rhs_vec[col_idx + 1], 
                       cos_vals[col_idx], sin_vals[col_idx]);
}

double computeVectorNorm(const double *vector, uint length) {
    double sum = 0.0;
    double compensation = 0.0;
    for (uint i = 0; i < length; ++i) {
        double y = vector[i] * vector[i] - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    return std::sqrt(sum);
}

void csrMatrixVectorProduct(const uint *row_offsets, const uint *col_indices, 
                           const double *matrix_values, const double *input_vec, 
                           double *output_vec, uint num_rows) {
    for (uint row_id = 0; row_id < num_rows; ++row_id) {
        uint row_begin = row_offsets[row_id];
        uint row_finish = row_offsets[row_id + 1];
        double accumulator = 0.0;
        for (uint idx = row_begin; idx < row_finish; ++idx) {
            accumulator += matrix_values[idx] * input_vec[col_indices[idx]];
        }
        output_vec[row_id] = accumulator;
    }
}

double vectorDotProduct(const double *vec_a, const double *vec_b, uint dimension) {
    constexpr int CHUNK = 4;
    double partial_sums[CHUNK] = {0.0, 0.0, 0.0, 0.0};
    uint i = 0;
    for (; i + CHUNK <= dimension; i += CHUNK) {
        partial_sums[0] += vec_a[i] * vec_b[i];
        partial_sums[1] += vec_a[i+1] * vec_b[i+1];
        partial_sums[2] += vec_a[i+2] * vec_b[i+2];
        partial_sums[3] += vec_a[i+3] * vec_b[i+3];
    }
    for (; i < dimension; ++i) {
        partial_sums[0] += vec_a[i] * vec_b[i];
    }
    return partial_sums[0] + partial_sums[1] + partial_sums[2] + partial_sums[3];
}

void vectorAxpy(double scalar, const double *x_vec, double *y_vec, uint length) {
    for (uint i = 0; i < length; ++i)
        y_vec[i] += scalar * x_vec[i];
}

void vectorScale(double scalar, double *vector, uint length) {
    for (uint i = 0; i < length; ++i)
        vector[i] *= scalar;
}

void vectorCopy(const double *source, double *destination, uint length) {
    memcpy(destination, source, length * sizeof(double));
}

void solveUpperTriangular(int matrix_dim, int last_col, double *upper_matrix, double *solution) {
    for (int col = last_col; col >= 0; col--) {
        double diag_elem = upper_matrix[matrix_dim * col + col];
        solution[col] /= diag_elem;
        double sol_val = solution[col];
        for (int row = 0; row < col; row++) {
            solution[row] -= upper_matrix[row * matrix_dim + col] * sol_val;
        }
    }
}

// ============ Device Kernels ============

__device__ inline double wavefront_butterfly_reduce(double value) {
    for (int delta = (WAVEFRONT_SIZE >> 1); delta > 0; delta >>= 1) {
        value += __shfl_xor(value, delta, WAVEFRONT_SIZE);
    }
    return value;
}

__global__ void convert_fp64_to_fp32(int count, const double *__restrict__ input_fp64, 
                                     float *__restrict__ output_fp32) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = global_tid; i < count; i += stride) {
        output_fp32[i] = (float)input_fp64[i];
    }
}

__global__ void convert_and_scale_kernel(int count, const double *__restrict__ src, 
                                         double scale_factor, float *__restrict__ dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = (float)(src[idx] * scale_factor);
    }
}

__global__ void scale_copy_fp32(int count, const float *__restrict__ src, 
                                float multiplier, float *__restrict__ dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = src[idx] * multiplier;
    }
}

__global__ void tree_reduction_kernel(const double *__restrict__ partial_results, 
                                     int num_partials, double *__restrict__ final_result) {
    __shared__ double shared_buffer[BLOCK_SIZE];
    int tid = threadIdx.x;
    
    double thread_sum = 0.0;
    for (int i = tid; i < num_partials; i += blockDim.x) {
        thread_sum += partial_results[i];
    }
    shared_buffer[tid] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_buffer[tid] += shared_buffer[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) final_result[0] = shared_buffer[0];
}

__global__ void batched_inner_products_mixed_precision(
    int vec_length, int num_products, 
    const float *__restrict__ vec_w, 
    const float *__restrict__ basis_matrix, int leading_dim,
    double *__restrict__ partial_products) {
    
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    
    double accumulators[MAX_BASIS_VECTORS];
    #pragma unroll
    for (int j = 0; j < MAX_BASIS_VECTORS; ++j) {
        accumulators[j] = (j < num_products) ? 0.0 : 0.0;
    }
    
    for (int idx = block_id * blockDim.x + tid; idx < vec_length; 
         idx += num_blocks * blockDim.x) {
        
        double w_elem = (double)vec_w[idx];
        
        #pragma unroll
        for (int j = 0; j < num_products; ++j) {
            double basis_elem = (double)basis_matrix[idx + j * leading_dim];
            accumulators[j] += w_elem * basis_elem;
        }
    }
    
    __shared__ double shared_mem[BLOCK_SIZE];
    for (int j = 0; j < num_products; ++j) {
        shared_mem[tid] = accumulators[j];
        __syncthreads();
        
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_mem[tid] += shared_mem[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            partial_products[block_id * num_products + j] = shared_mem[0];
        }
        __syncthreads();
    }
}

__global__ void multi_result_reduction(const double *__restrict__ partials, 
                                       int num_results, int num_blocks,
                                       double *__restrict__ outputs) {
    __shared__ double reduction_buffer[BLOCK_SIZE];
    int tid = threadIdx.x;
    
    for (int result_id = 0; result_id < num_results; ++result_id) {
        double local_sum = 0.0;
        
        for (int block_idx = tid; block_idx < num_blocks; block_idx += blockDim.x) {
            local_sum += partials[block_idx * num_results + result_id];
        }
        
        reduction_buffer[tid] = local_sum;
        __syncthreads();
        
        for (int step = BLOCK_SIZE >> 1; step > 0; step >>= 1) {
            if (tid < step) {
                reduction_buffer[tid] += reduction_buffer[tid + step];
            }
            __syncthreads();
        }
        
        if (tid == 0) outputs[result_id] = reduction_buffer[0];
        __syncthreads();
    }
}

__global__ void orthogonal_projection_with_norm(
    int dimension, int num_basis, 
    const float *__restrict__ basis_vecs, int ld,
    const double *__restrict__ coeffs,
    float *__restrict__ residual,
    double *__restrict__ norm_partials) {
    
    __shared__ double shmem_buffer[BLOCK_SIZE];
    int tid = threadIdx.x;
    int elem_id = blockIdx.x * blockDim.x + tid;
    
    double squared_residual = 0.0;
    
    if (elem_id < dimension) {
        double projection = 0.0;
        for (int basis_idx = 0; basis_idx < num_basis; ++basis_idx) {
            float basis_val = basis_vecs[elem_id + basis_idx * ld];
            projection += (double)basis_val * coeffs[basis_idx];
        }
        
        float updated_res = residual[elem_id] - (float)projection;
        residual[elem_id] = updated_res;
        squared_residual = (double)updated_res * (double)updated_res;
    }
    
    shmem_buffer[tid] = squared_residual;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem_buffer[tid] += shmem_buffer[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) norm_partials[blockIdx.x] = shmem_buffer[0];
}

__global__ void update_solution_linear_combination(
    int vec_size, int num_terms,
    const float *__restrict__ basis_matrix, int ld,
    const double *__restrict__ weights,
    double *__restrict__ solution) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vec_size) return;
    
    double combination = 0.0;
    for (int term = 0; term < num_terms; ++term) {
        float basis_elem = basis_matrix[idx + term * ld];
        combination += (double)basis_elem * weights[term];
    }
    
    solution[idx] += combination;
}

// SpMV kernels - 保持原有实现
template<typename IndexType, unsigned int VEC_PER_BLK, unsigned int THR_PER_VEC>
__global__ void spmv_vector_minus_b_partial(
    const IndexType row_num, const IndexType *__restrict__ rowPtr,
    const IndexType *__restrict__ colInd, const double *__restrict__ vals,
    const double *__restrict__ x, const double *__restrict__ b,
    double *__restrict__ r, double *__restrict__ partial) {

    const IndexType tid = VEC_PER_BLK * THR_PER_VEC * blockIdx.x + threadIdx.x;
    const IndexType lane = threadIdx.x & (THR_PER_VEC - 1);
    const IndexType v_id = threadIdx.x / THR_PER_VEC;
    const IndexType row = tid / THR_PER_VEC;

    __shared__ double s_r2[VEC_PER_BLK];
    if (lane == 0) s_r2[v_id] = 0.0;

    if (row < row_num) {
        IndexType row_start = rowPtr[row];
        IndexType row_end = rowPtr[row + 1];

        double br = 0.0;
        if (lane == 0) br = b[row];

        double sum = 0.0;
        for (IndexType jj = row_start + lane; jj < row_end; jj += THR_PER_VEC) {
            sum += vals[jj] * x[colInd[jj]];
        }

        for (unsigned off = THR_PER_VEC >> 1; off > 0; off >>= 1)
            sum += __shfl_down(sum, off, THR_PER_VEC);

        if (lane == 0) {
            double ri = sum - br;
            r[row] = ri;
            s_r2[v_id] = ri * ri;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double blk = 0.0;
        #pragma unroll
        for (unsigned v = 0; v < VEC_PER_BLK; ++v)
            blk += s_r2[v];
        partial[blockIdx.x] = blk;
    }
}

template<typename IndexType, unsigned int VEC_PER_BLK, unsigned int THR_PER_VEC>
__global__ void spmv_vector_f32(
    const IndexType row_num, const IndexType *__restrict__ rowPtr,
    const IndexType *__restrict__ colInd, const float *__restrict__ vals,
    const float *__restrict__ x, float *__restrict__ y) {

    const IndexType tid = VEC_PER_BLK * THR_PER_VEC * blockIdx.x + threadIdx.x;
    const IndexType lane = threadIdx.x & (THR_PER_VEC - 1);
    const IndexType row = tid / THR_PER_VEC;

    if (row >= row_num) return;

    IndexType row_start = rowPtr[row];
    IndexType row_end = rowPtr[row + 1];

    float sum = 0.0f;
    for (IndexType jj = row_start + lane; jj < row_end; jj += THR_PER_VEC) {
        sum += vals[jj] * x[colInd[jj]];
    }

    for (unsigned off = THR_PER_VEC >> 1; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, THR_PER_VEC);

    if (lane == 0) y[row] = sum;
}

// ============ Helper Functions ============
static inline void perform_batched_inner_products(
    int vec_len, int num_prods, 
    const float *dev_vec_w, const float *dev_basis, int basis_ld,
    double *dev_partials, double *dev_results) {
    
    int grid_size = compute_reduction_grid(vec_len);
    batched_inner_products_mixed_precision<<<grid_size, BLOCK_SIZE>>>(
        vec_len, num_prods, dev_vec_w, dev_basis, basis_ld, dev_partials);
    multi_result_reduction<<<1, BLOCK_SIZE>>>(dev_partials, num_prods, grid_size, dev_results);
}

// ============ GMRES主函数 ============
RESULT gmres(SpM<double> *A_d, double *x_d, double *_b) {
    const uint matrix_size = A_d->nrows;
    const uint nonzeros = A_d->rows[matrix_size];

    //==================以下代码禁止修改=================
    double init_res = 0.0;
    for (uint i = 0; i < matrix_size; ++i)
        init_res += _b[i] * _b[i];
    init_res = std::sqrt(init_res);
    double RESID_LIMIT = REL_RESID_LIMIT * init_res;
    //==================以上代码禁止修改=================

    const int restart_dim = RESTART_TIMES;

    // Host端数据
    std::vector<double> hessenberg_matrix((restart_dim + 1) * restart_dim, 0.0);
    std::vector<double> givens_cos(restart_dim, 0.0), givens_sin(restart_dim, 0.0);
    std::vector<double> rhs_vector(restart_dim + 1, 0.0);
    std::vector<double> host_coefficients(MAX_BASIS_VECTORS, 0.0);

    // Device内存
    unsigned *dev_row_offsets = nullptr, *dev_col_indices = nullptr;
    double *dev_matrix_vals = nullptr, *dev_solution = nullptr, *dev_rhs = nullptr, *dev_residual = nullptr;
    float *dev_matrix_vals_sp = nullptr, *dev_krylov_basis = nullptr, *dev_work_vector = nullptr;
    double *dev_reduction_buffer = nullptr, *dev_scalar_result = nullptr;
    double *dev_batch_partials = nullptr, *dev_hessenberg_col = nullptr, *dev_coeffs = nullptr;
    double *dev_norm_partials = nullptr, *dev_spmv_partials = nullptr;

    CHECK_HIP_ERROR(hipMalloc(&dev_row_offsets, (matrix_size + 1) * sizeof(unsigned)));
    CHECK_HIP_ERROR(hipMalloc(&dev_col_indices, nonzeros * sizeof(unsigned)));
    CHECK_HIP_ERROR(hipMalloc(&dev_matrix_vals, nonzeros * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_matrix_vals_sp, nonzeros * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dev_solution, matrix_size * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_rhs, matrix_size * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_residual, matrix_size * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_krylov_basis, (restart_dim + 1) * (size_t)matrix_size * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&dev_work_vector, matrix_size * sizeof(float)));

    int max_reduction_blocks = compute_reduction_grid(matrix_size);
    CHECK_HIP_ERROR(hipMalloc(&dev_reduction_buffer, max_reduction_blocks * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_scalar_result, sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_batch_partials, max_reduction_blocks * MAX_BASIS_VECTORS * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_hessenberg_col, MAX_BASIS_VECTORS * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_coeffs, restart_dim * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_norm_partials, compute_grid_size(matrix_size) * sizeof(double)));
    CHECK_HIP_ERROR(hipMalloc(&dev_spmv_partials, ((matrix_size + 4 - 1) / 4) * sizeof(double)));

    // H2D传输
    CHECK_HIP_ERROR(hipMemcpy(dev_row_offsets, A_d->rows, (matrix_size + 1) * sizeof(unsigned), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dev_col_indices, A_d->cols, nonzeros * sizeof(unsigned), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dev_matrix_vals, A_d->vals, nonzeros * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dev_solution, x_d, matrix_size * sizeof(double), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dev_rhs, _b, matrix_size * sizeof(double), hipMemcpyHostToDevice));

    // =================== Warmup ===================
    {
        convert_fp64_to_fp32<<<compute_grid_size(nonzeros), BLOCK_SIZE>>>(nonzeros, dev_matrix_vals, dev_matrix_vals_sp);
        constexpr unsigned VEC_PER_BLK = 4, THR_PER_VEC = 16;
        dim3 blk(VEC_PER_BLK * THR_PER_VEC), grd((matrix_size + VEC_PER_BLK - 1) / VEC_PER_BLK);
        spmv_vector_minus_b_partial<uint, VEC_PER_BLK, THR_PER_VEC><<<grd, blk>>>(
            matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        spmv_vector_f32<uint, VEC_PER_BLK, THR_PER_VEC><<<grd, blk>>>(
            matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals_sp, dev_krylov_basis, dev_work_vector);
        CHECK_HIP_ERROR(hipDeviceSynchronize());
    }

    // 计时
    hipEvent_t event_start, event_stop;
    CHECK_HIP_ERROR(hipEventCreate(&event_start));
    CHECK_HIP_ERROR(hipEventCreate(&event_stop));
    CHECK_HIP_ERROR(hipEventRecord(event_start, 0));

    convert_fp64_to_fp32<<<compute_grid_size(nonzeros), BLOCK_SIZE>>>(nonzeros, dev_matrix_vals, dev_matrix_vals_sp);

    uint avg_row_nnz = std::max(1u, nonzeros / matrix_size);
    unsigned VEC_PER_BLK = 4, THR_PER_VEC = 16;
    if (avg_row_nnz <= 16) THR_PER_VEC = 4;
    else if (avg_row_nnz <= 32) THR_PER_VEC = 8;
    else if (avg_row_nnz <= 64) THR_PER_VEC = 16;
    else THR_PER_VEC = 32;

    int total_iterations = 0;
    double current_residual = 0.0;

    // ====== GMRES 主循环 ======
    do {
        double residual_norm_sq = 0.0;
        dim3 outer_block(VEC_PER_BLK * THR_PER_VEC), outer_grid((matrix_size + VEC_PER_BLK - 1) / VEC_PER_BLK);

        if (THR_PER_VEC == 4) {
            spmv_vector_minus_b_partial<uint, 4, 4><<<outer_grid, outer_block>>>(
                matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        } else if (THR_PER_VEC == 8) {
            spmv_vector_minus_b_partial<uint, 4, 8><<<outer_grid, outer_block>>>(
                matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        } else if (THR_PER_VEC == 16) {
            spmv_vector_minus_b_partial<uint, 4, 16><<<outer_grid, outer_block>>>(
                matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        } else {
            spmv_vector_minus_b_partial<uint, 4, 32><<<outer_grid, outer_block>>>(
                matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        }

        tree_reduction_kernel<<<1, BLOCK_SIZE>>>(dev_spmv_partials, (matrix_size + VEC_PER_BLK - 1) / VEC_PER_BLK, dev_scalar_result);
        CHECK_HIP_ERROR(hipMemcpy(&residual_norm_sq, dev_scalar_result, sizeof(double), hipMemcpyDeviceToHost));

        if (!std::isfinite(residual_norm_sq) || residual_norm_sq < 0.0) residual_norm_sq = 0.0;
        double norm_r = std::sqrt(residual_norm_sq);

        convert_and_scale_kernel<<<compute_grid_size(matrix_size), BLOCK_SIZE>>>(
            matrix_size, dev_residual, (norm_r == 0.0 ? 0.0 : -1.0/norm_r), dev_krylov_basis + 0);

        std::fill(rhs_vector.begin(), rhs_vector.end(), 0.0);
        rhs_vector[0] = norm_r;
        current_residual = fabs(norm_r);

        if (current_residual <= RESID_LIMIT || total_iterations >= ITERATION_LIMIT) break;

        int inner_iter = -1;
        do {
            inner_iter++;
            total_iterations++;

            if (THR_PER_VEC == 4) {
                spmv_vector_f32<uint, 4, 4><<<outer_grid, outer_block>>>(
                    matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals_sp, 
                    dev_krylov_basis + (size_t)inner_iter * matrix_size, dev_work_vector);
            } else if (THR_PER_VEC == 8) {
                spmv_vector_f32<uint, 4, 8><<<outer_grid, outer_block>>>(
                    matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals_sp, 
                    dev_krylov_basis + (size_t)inner_iter * matrix_size, dev_work_vector);
            } else if (THR_PER_VEC == 16) {
                spmv_vector_f32<uint, 4, 16><<<outer_grid, outer_block>>>(
                    matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals_sp, 
                    dev_krylov_basis + (size_t)inner_iter * matrix_size, dev_work_vector);
            } else {
                spmv_vector_f32<uint, 4, 32><<<outer_grid, outer_block>>>(
                    matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals_sp, 
                    dev_krylov_basis + (size_t)inner_iter * matrix_size, dev_work_vector);
            }

            perform_batched_inner_products(matrix_size, inner_iter + 1, dev_work_vector, dev_krylov_basis, 
                                          matrix_size, dev_batch_partials, dev_hessenberg_col);
            CHECK_HIP_ERROR(hipMemcpy(host_coefficients.data(), dev_hessenberg_col, (inner_iter + 1) * sizeof(double), hipMemcpyDeviceToHost));

            for (int k = 0; k <= inner_iter; ++k) {
                hessenberg_matrix[k * restart_dim + inner_iter] = host_coefficients[k];
            }

            orthogonal_projection_with_norm<<<compute_grid_size(matrix_size), BLOCK_SIZE>>>(
                matrix_size, inner_iter + 1, dev_krylov_basis, matrix_size, dev_hessenberg_col, dev_work_vector, dev_norm_partials);
            tree_reduction_kernel<<<1, BLOCK_SIZE>>>(dev_norm_partials, compute_grid_size(matrix_size), dev_scalar_result);

            double norm_sq = 0.0;
            CHECK_HIP_ERROR(hipMemcpy(&norm_sq, dev_scalar_result, sizeof(double), hipMemcpyDeviceToHost));
            if (!std::isfinite(norm_sq) || norm_sq < 0.0) norm_sq = 0.0;
            double h_next = std::sqrt(norm_sq);
            hessenberg_matrix[(inner_iter + 1) * restart_dim + inner_iter] = h_next;

            if (h_next != 0.0) {
                float scale_factor = (float)(1.0 / h_next);
                scale_copy_fp32<<<compute_grid_size(matrix_size), BLOCK_SIZE>>>(
                    matrix_size, dev_work_vector, scale_factor, dev_krylov_basis + (size_t)(inner_iter + 1) * matrix_size);
            } else {
                CHECK_HIP_ERROR(hipMemcpy(dev_krylov_basis + (size_t)(inner_iter + 1) * matrix_size, dev_work_vector, 
                                         matrix_size * sizeof(float), hipMemcpyDeviceToDevice));
            }

            performGivensSequence(restart_dim, hessenberg_matrix.data(), givens_cos.data(), 
                                 givens_sin.data(), rhs_vector.data(), inner_iter);

            if (total_iterations >= ITERATION_LIMIT) break;

        } while (inner_iter + 1 < restart_dim && total_iterations < ITERATION_LIMIT);

        solveUpperTriangular(restart_dim, inner_iter, hessenberg_matrix.data(), rhs_vector.data());

        CHECK_HIP_ERROR(hipMemcpy(dev_coeffs, rhs_vector.data(), (inner_iter + 1) * sizeof(double), hipMemcpyHostToDevice));
        update_solution_linear_combination<<<compute_grid_size(matrix_size), BLOCK_SIZE>>>(
            matrix_size, inner_iter + 1, dev_krylov_basis, matrix_size, dev_coeffs, dev_solution);

        if (THR_PER_VEC == 4) {
            spmv_vector_minus_b_partial<uint, 4, 4><<<outer_grid, outer_block>>>(
                matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        } else if (THR_PER_VEC == 8) {
            spmv_vector_minus_b_partial<uint, 4, 8><<<outer_grid, outer_block>>>(
                matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        } else if (THR_PER_VEC == 16) {
            spmv_vector_minus_b_partial<uint, 4, 16><<<outer_grid, outer_block>>>(
                matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        } else {
            spmv_vector_minus_b_partial<uint, 4, 32><<<outer_grid, outer_block>>>(
                matrix_size, dev_row_offsets, dev_col_indices, dev_matrix_vals, dev_solution, dev_rhs, dev_residual, dev_spmv_partials);
        }
        tree_reduction_kernel<<<1, BLOCK_SIZE>>>(dev_spmv_partials, (matrix_size + VEC_PER_BLK - 1) / VEC_PER_BLK, dev_scalar_result);
        double final_norm_sq = 0.0;
        CHECK_HIP_ERROR(hipMemcpy(&final_norm_sq, dev_scalar_result, sizeof(double), hipMemcpyDeviceToHost));
        if (!std::isfinite(final_norm_sq) || final_norm_sq < 0.0) final_norm_sq = 0.0;
        current_residual = std::sqrt(final_norm_sq);

    } while (current_residual > RESID_LIMIT && total_iterations <= ITERATION_LIMIT);

    CHECK_HIP_ERROR(hipEventRecord(event_stop, 0));
    CHECK_HIP_ERROR(hipEventSynchronize(event_stop));
    float elapsed_time = 0.0;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsed_time, event_start, event_stop));

    CHECK_HIP_ERROR(hipMemcpy(x_d, dev_solution, matrix_size * sizeof(double), hipMemcpyDeviceToHost));

    // 清理
    CHECK_HIP_ERROR(hipFree(dev_row_offsets));
    CHECK_HIP_ERROR(hipFree(dev_col_indices));
    CHECK_HIP_ERROR(hipFree(dev_matrix_vals));
    CHECK_HIP_ERROR(hipFree(dev_matrix_vals_sp));
    CHECK_HIP_ERROR(hipFree(dev_solution));
    CHECK_HIP_ERROR(hipFree(dev_rhs));
    CHECK_HIP_ERROR(hipFree(dev_residual));
    CHECK_HIP_ERROR(hipFree(dev_krylov_basis));
    CHECK_HIP_ERROR(hipFree(dev_work_vector));
    CHECK_HIP_ERROR(hipFree(dev_reduction_buffer));
    CHECK_HIP_ERROR(hipFree(dev_scalar_result));
    CHECK_HIP_ERROR(hipFree(dev_batch_partials));
    CHECK_HIP_ERROR(hipFree(dev_hessenberg_col));
    CHECK_HIP_ERROR(hipFree(dev_coeffs));
    CHECK_HIP_ERROR(hipFree(dev_norm_partials));
    CHECK_HIP_ERROR(hipFree(dev_spmv_partials));
    CHECK_HIP_ERROR(hipEventDestroy(event_start));
    CHECK_HIP_ERROR(hipEventDestroy(event_stop));

    return make_tuple(total_iterations, elapsed_time, current_residual / init_res);
}

// 这些CPU函数供main.cpp调用（保持接口兼容）
double calculateNorm(const double *vec, uint N) {
    return computeVectorNorm(vec, N);
}

void spmv(const uint *rowPtr, const uint *colInd, const double *values,
          const double *x, double *y, uint numRows) {
    csrMatrixVectorProduct(rowPtr, colInd, values, x, y, numRows);
}