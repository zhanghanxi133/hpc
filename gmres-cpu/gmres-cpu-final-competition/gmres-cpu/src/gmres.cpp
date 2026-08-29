#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <cassert>
#include <chrono>
#include "sparseMatrix.hpp"
#include "gmres.hpp"
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>

using namespace std;

const int RESTART_TIMES = 20;  // 禁止修改
const double REL_RESID_LIMIT = 1e-6; // 禁止修改
const int ITERATION_LIMIT = 10000; // 禁止修改

#define HIP_CHECK(cmd) \
do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIPBLAS_CHECK(cmd) \
do { \
    hipblasStatus_t status = cmd; \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "HIPBLAS error: " << hipblasStatusToString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIPSPARSE_CHECK(cmd) \
do { \
    hipsparseStatus_t status = cmd; \
    if (status != HIPSPARSE_STATUS_SUCCESS) { \
        std::cerr << "HIPSPARSE error: " << hipsparseGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ void apply_rotation_device_float(float *dx, float *dy, float cs, float sn) {
    float temp = cs * (*dx) + sn * (*dy);
    *dy = -sn * (*dx) + cs * (*dy);
    *dx = temp;
}

__device__ void generate_rotation_device_float(float *dx, float *dy, float *cs, float *sn) {
    if (*dy == 0.0f) {
        *cs = 1.0f;
        *sn = 0.0f;
    } else {
        float r = hypotf(*dx, *dy);
        *cs = *dx / r;
        *sn = *dy / r;
        *dx = r;
        *dy = 0.0f;
    }
}

__global__ void arnoldi_update_and_givens_float_kernel(float* d_H, float* d_cs, float* d_sn, float* d_s, int num_iters, int H_lda) {
    for (int i = 0; i < num_iters; i++) {
        float* H_col = d_H + i * H_lda;
        for (int k = 0; k < i; k++) {
            apply_rotation_device_float(&H_col[k], &H_col[k+1], d_cs[k], d_sn[k]);
        }
        generate_rotation_device_float(&H_col[i], &H_col[i+1], &d_cs[i], &d_sn[i]);
        apply_rotation_device_float(&d_s[i], &d_s[i+1], d_cs[i], d_sn[i]);
    }
}

__global__ void set_value_float_kernel(float* addr, float val) {
    *addr = val;
}

__global__ void find_convergence_and_resid_float_kernel(const float* d_s, int num_iters, float limit, int* d_final_i, float* d_resid_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int final_idx = num_iters - 1;
        for (int k = 0; k < num_iters; ++k) {
            if (fabsf(d_s[k + 1]) <= limit) {
                final_idx = k;
                break;
            }
        }
        *d_final_i = final_idx;
        *d_resid_out = fabsf(d_s[final_idx + 1]);
    }
}

// Kernel to cast from double to float
__global__ void cast_double_to_float_kernel(float* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)src[idx];
    }
}

// Kernel to cast from double to float and scale (r_float = r_double * scale)
__global__ void cast_and_scale_double_to_float_kernel(float* dst, const double* src, double scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)(src[idx] * scale);
    }
}

// Kernel to update solution: x_double = x_double + V_float * y_float
__global__ void update_solution_kernel(double* d_x, const float* d_V, const float* d_s, int N, int k, int ldv) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        double sum = 0.0;
        for (int j = 0; j <= k; ++j) {
            sum += (double)d_V[row + j * ldv] * (double)d_s[j];
        }
        d_x[row] += sum;
    }
}

void spmv_dcu(hipsparseHandle_t handle, hipsparseSpMatDescr_t matA, hipsparseDnVecDescr_t vecX, hipsparseDnVecDescr_t vecY, const void* alpha, const void* beta, hipDataType type, void* buffer) {
    HIPSPARSE_CHECK(hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 alpha, matA, vecX, beta, vecY,
                                 type, HIPSPARSE_SPMV_ALG_DEFAULT, buffer));
}

RESULT gmres(SpM<double> *A_d, double *x_d, double *_b)
{
    const uint N = A_d->nrows;
    const int m = RESTART_TIMES;
    const int H_lda = m + 1;

    hipblasHandle_t hipblasHandle;
    hipsparseHandle_t hipsparseHandle;
    HIPBLAS_CHECK(hipblasCreate(&hipblasHandle));
    HIPSPARSE_CHECK(hipsparseCreate(&hipsparseHandle));

    // Double precision variables (outer loop)
    double *d_x, *d_b, *d_r;
    uint   *d_rows, *d_cols;
    double *d_vals;

    // Float precision variables (inner loop)
    float *d_V_float, *d_H_float, *d_s_float, *d_cs_float, *d_sn_float;
    float *d_w_float, *d_vals_float;

    // Utility
    int* d_final_i;
    float* d_resid_val_float;

    // Double
    HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_r, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_rows, (A_d->nrows + 1) * sizeof(uint)));
    HIP_CHECK(hipMalloc(&d_cols, A_d->nnz * sizeof(uint)));
    HIP_CHECK(hipMalloc(&d_vals, A_d->nnz * sizeof(double)));

    // Float
    HIP_CHECK(hipMalloc(&d_V_float, N * (m + 1) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_H_float, H_lda * m * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_s_float, (m + 1) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cs_float, m * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sn_float, m * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w_float, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_vals_float, A_d->nnz * sizeof(float)));

    // Utility
    HIP_CHECK(hipMalloc(&d_final_i, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_resid_val_float, sizeof(float)));

    // Host -> Device Data Copy
    HIP_CHECK(hipMemcpy(d_x, x_d, N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, _b, N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rows, A_d->rows, (A_d->nrows + 1) * sizeof(uint), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cols, A_d->cols, A_d->nnz * sizeof(uint), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vals, A_d->vals, A_d->nnz * sizeof(double), hipMemcpyHostToDevice));

    // Create hipSPARSE Descriptors (for both double and float)
    hipsparseSpMatDescr_t matA_descr_double, matA_descr_float;
    hipsparseDnVecDescr_t vecX_descr_double, vecR_descr_double;
    hipsparseDnVecDescr_t vecW_descr_float;
    std::vector<hipsparseDnVecDescr_t> vecV_descr_float(m + 1);

    // Double descriptors for outer loop
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA_descr_double, N, N, A_d->nnz, d_rows, d_cols, d_vals, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecX_descr_double, N, d_x, HIP_R_64F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecR_descr_double, N, d_r, HIP_R_64F));

    // Float descriptors for inner loop
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA_descr_float, N, N, A_d->nnz, d_rows, d_cols, d_vals_float, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecW_descr_float, N, d_w_float, HIP_R_32F));
    for (int i = 0; i <= m; ++i) {
        HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecV_descr_float[i], N, d_V_float + i * N, HIP_R_32F));
    }
    
    // Allocate SpMV buffers (for both precisions)
    void* d_spmv_buffer_double = nullptr, *d_spmv_buffer_float = nullptr;
    size_t bufferSize_d = 0, bufferSize_f = 0;
    const double alpha_d_one = 1.0, beta_d_zero = 0.0;
    const float alpha_f_one = 1.0f, beta_f_zero = 0.0f;
    HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha_d_one, matA_descr_double, vecX_descr_double, &beta_d_zero, vecR_descr_double, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize_d));
    HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha_f_one, matA_descr_float, vecV_descr_float[0], &beta_f_zero, vecW_descr_float, HIP_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize_f));
    HIP_CHECK(hipMalloc(&d_spmv_buffer_double, bufferSize_d));
    HIP_CHECK(hipMalloc(&d_spmv_buffer_float, bufferSize_f));

    // Initialization
    double initial_b_norm = 0.0;
    HIPBLAS_CHECK(hipblasDnrm2(hipblasHandle, N, d_b, 1, &initial_b_norm));
    if (initial_b_norm == 0.0) {
         HIP_CHECK(hipMemset(x_d, 0, N*sizeof(double)));
         return make_tuple(0, 0.0f, 0.0);
    }
    const double RESID_LIMIT = REL_RESID_LIMIT * initial_b_norm;
    double resid = initial_b_norm;
    int iteration = 0;
    
    const double alpha_one_d = 1.0, alpha_neg_one_d = -1.0;
    const float  alpha_one_f = 1.0f, alpha_neg_one_f = -1.0f, alpha_zero_f = 0.0f;

    // Warm-up
    {
        dim3 block_warmup(256);
        dim3 grid_cast_warmup((A_d->nnz + block_warmup.x - 1) / block_warmup.x);
        hipLaunchKernelGGL(cast_double_to_float_kernel, grid_cast_warmup, block_warmup, 0, 0, d_vals_float, d_vals, A_d->nnz);
        dim3 grid_N_warmup((N + block_warmup.x - 1) / block_warmup.x);

        // 2. 外循环预热 (Double)
        HIP_CHECK(hipMemcpy(d_r, d_b, N * sizeof(double), hipMemcpyDeviceToDevice));
        spmv_dcu(hipsparseHandle, matA_descr_double, vecX_descr_double, vecR_descr_double, &alpha_neg_one_d, &alpha_one_d, HIP_R_64F, d_spmv_buffer_double);
        double dummy_beta = 0.0;
        HIPBLAS_CHECK(hipblasDnrm2(hipblasHandle, N, d_r, 1, &dummy_beta));

        // 3. 精度转换Kernel预热
        hipLaunchKernelGGL(cast_and_scale_double_to_float_kernel, grid_N_warmup, block_warmup, 0, 0, d_V_float, d_r, 1.0, N);
        hipLaunchKernelGGL(set_value_float_kernel, dim3(1), dim3(1), 0, 0, d_s_float, 1.0f);

        // 4. 内循环预热 (Float)
        spmv_dcu(hipsparseHandle, matA_descr_float, vecV_descr_float[0], vecW_descr_float, &alpha_one_f, &alpha_zero_f, HIP_R_32F, d_spmv_buffer_float);
        HIPBLAS_CHECK(hipblasSgemv(hipblasHandle, HIPBLAS_OP_T, N, 1, &alpha_one_f, d_V_float, N, d_w_float, 1, &alpha_zero_f, d_H_float, 1));
        HIPBLAS_CHECK(hipblasSgemv(hipblasHandle, HIPBLAS_OP_N, N, 1, &alpha_neg_one_f, d_V_float, N, d_H_float, 1, &alpha_one_f, d_w_float, 1));
        float dummy_h = 0.0f;
        HIPBLAS_CHECK(hipblasSnrm2(hipblasHandle, N, d_w_float, 1, &dummy_h));
        float dummy_inv_h = 1.0f;
        HIPBLAS_CHECK(hipblasSscal(hipblasHandle, N, &dummy_inv_h, d_w_float, 1));

        // 5. 后处理Kernel预热
        hipLaunchKernelGGL(arnoldi_update_and_givens_float_kernel, dim3(1), dim3(1), 0, 0, d_H_float, d_cs_float, d_sn_float, d_s_float, 1, H_lda);
        hipLaunchKernelGGL(find_convergence_and_resid_float_kernel, dim3(1), dim3(1), 0, 0, d_s_float, 1, (float)RESID_LIMIT, d_final_i, d_resid_val_float);
        HIPBLAS_CHECK(hipblasStrsv(hipblasHandle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, 1, d_H_float, H_lda, d_s_float, 1));
        hipLaunchKernelGGL(update_solution_kernel, grid_N_warmup, block_warmup, 0, 0, d_x, d_V_float, d_s_float, N, 0, N);

        HIP_CHECK(hipDeviceSynchronize());
    }

    // =======DCU计时开始==========
    hipEvent_t test_start_event, test_stop_event;
    HIP_CHECK(hipEventCreate(&test_start_event));
    HIP_CHECK(hipEventCreate(&test_stop_event));
    HIP_CHECK(hipEventRecord(test_start_event, 0));

    dim3 block(256);
    dim3 grid_cast((A_d->nnz + block.x - 1) / block.x);
    hipLaunchKernelGGL(cast_double_to_float_kernel, grid_cast, block, 0, 0, d_vals_float, d_vals, A_d->nnz);

    do {
        // ========== 外迭代 (Double Precision) ============
        // r = b - A*x
        HIP_CHECK(hipMemcpy(d_r, d_b, N * sizeof(double), hipMemcpyDeviceToDevice));
        spmv_dcu(hipsparseHandle, matA_descr_double, vecX_descr_double, vecR_descr_double, &alpha_neg_one_d, &alpha_one_d, HIP_R_64F, d_spmv_buffer_double);
        
        double beta = 0.0;
        HIPBLAS_CHECK(hipblasDnrm2(hipblasHandle, N, d_r, 1, &beta));
        
        // ========== PRECISION TRANSITION: Double -> Float ==========
        // v_0_float = r_double / beta
        double inv_beta = (beta == 0.0) ? 1.0 : 1.0 / beta;
        dim3 grid_N((N + block.x - 1) / block.x);
        hipLaunchKernelGGL(cast_and_scale_double_to_float_kernel, grid_N, block, 0, 0, d_V_float, d_r, inv_beta, N);

        // s_float[0] = beta
        HIP_CHECK(hipMemset(d_s_float, 0, (m + 1) * sizeof(float)));
        hipLaunchKernelGGL(set_value_float_kernel, dim3(1), dim3(1), 0, 0, d_s_float, (float)beta);

        resid = beta;
        if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) break;
        
        int inner_loop_actual_iters = 0;

        // ========== 内迭代 (Float Precision) ============
        for (int i = 0; i < m; ++i) {
            iteration++;
            inner_loop_actual_iters = i + 1;
            
            // w = A * v_i
            spmv_dcu(hipsparseHandle, matA_descr_float, vecV_descr_float[i], vecW_descr_float, &alpha_one_f, &alpha_zero_f, HIP_R_32F, d_spmv_buffer_float);
            
            float* d_Hi_col = d_H_float + i * H_lda;
            
            // Modified Gram-Schmidt: h_col = V^T * w
            HIPBLAS_CHECK(hipblasSgemv(hipblasHandle, HIPBLAS_OP_T, N, i + 1, &alpha_one_f, d_V_float, N, d_w_float, 1, &alpha_zero_f, d_Hi_col, 1));
            // w = w - V * h_col
            HIPBLAS_CHECK(hipblasSgemv(hipblasHandle, HIPBLAS_OP_N, N, i + 1, &alpha_neg_one_f, d_V_float, N, d_Hi_col, 1, &alpha_one_f, d_w_float, 1));

            float h_i1_i = 0.0f;
            HIPBLAS_CHECK(hipblasSnrm2(hipblasHandle, N, d_w_float, 1, &h_i1_i));
            hipLaunchKernelGGL(set_value_float_kernel, dim3(1), dim3(1), 0, 0, d_Hi_col + i + 1, h_i1_i);
            
            if (h_i1_i != 0.0f){
                float inv_h = 1.0f / h_i1_i;
                HIPBLAS_CHECK(hipblasSscal(hipblasHandle, N, &inv_h, d_w_float, 1));
                HIP_CHECK(hipMemcpy(d_V_float + (i + 1) * N, d_w_float, N * sizeof(float), hipMemcpyDeviceToDevice));
            } else {
                break;
            }

            if(iteration >= ITERATION_LIMIT) break;
        }

        if (inner_loop_actual_iters > 0) {
            hipLaunchKernelGGL(arnoldi_update_and_givens_float_kernel, dim3(1), dim3(1), 0, 0, 
                               d_H_float, d_cs_float, d_sn_float, d_s_float, inner_loop_actual_iters, H_lda);
        }

        // Convergence Check on Device
        int h_final_i;
        float resid_float;
        hipLaunchKernelGGL(find_convergence_and_resid_float_kernel, dim3(1), dim3(1), 0, 0,
                           d_s_float, inner_loop_actual_iters, (float)RESID_LIMIT, d_final_i, d_resid_val_float);
        HIP_CHECK(hipMemcpy(&h_final_i, d_final_i, sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&resid_float, d_resid_val_float, sizeof(float), hipMemcpyDeviceToHost));
        resid = (double)resid_float; // Update host residual in double

        bool converged_in_inner_loop = (h_final_i < inner_loop_actual_iters - 1);

        // Solve upper triangular system Hy=s (Float)
        if (h_final_i >= 0) {
            HIPBLAS_CHECK(hipblasStrsv(hipblasHandle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT,
                                      h_final_i + 1, d_H_float, H_lda, d_s_float, 1));
            
            // --- Update solution: x = x + V*y (PRECISION TRANSITION: Float -> Double) ---
            // Fused Kernel: x_double += (double)(V_float * y_float)
            hipLaunchKernelGGL(update_solution_kernel, grid_N, block, 0, 0,
                               d_x, d_V_float, d_s_float, N, h_final_i, N);
        }
        
        if (converged_in_inner_loop) break;

    } while (resid > RESID_LIMIT && iteration < ITERATION_LIMIT);

    // =======DCU计时结束==========
    HIP_CHECK(hipEventRecord(test_stop_event, 0));
    HIP_CHECK(hipEventSynchronize(test_stop_event));
    float test_time = 0.0;
    HIP_CHECK(hipEventElapsedTime(&test_time, test_start_event, test_stop_event));

    // Device -> Host Result Copy
    HIP_CHECK(hipMemcpy(x_d, d_x, N * sizeof(double), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_rows));
    HIP_CHECK(hipFree(d_cols));
    HIP_CHECK(hipFree(d_vals));
    HIP_CHECK(hipFree(d_V_float));
    HIP_CHECK(hipFree(d_H_float));
    HIP_CHECK(hipFree(d_s_float));
    HIP_CHECK(hipFree(d_cs_float));
    HIP_CHECK(hipFree(d_sn_float));
    HIP_CHECK(hipFree(d_w_float));
    HIP_CHECK(hipFree(d_vals_float));
    HIP_CHECK(hipFree(d_final_i));
    HIP_CHECK(hipFree(d_resid_val_float));
    HIP_CHECK(hipFree(d_spmv_buffer_double));
    HIP_CHECK(hipFree(d_spmv_buffer_float));

    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA_descr_double));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA_descr_float));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecX_descr_double));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecR_descr_double));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecW_descr_float));
    for (int i = 0; i <= m; ++i) { HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecV_descr_float[i])); }
    
    HIPBLAS_CHECK(hipblasDestroy(hipblasHandle));
    HIPSPARSE_CHECK(hipsparseDestroy(hipsparseHandle));
    HIP_CHECK(hipEventDestroy(test_start_event));
    HIP_CHECK(hipEventDestroy(test_stop_event));

    return make_tuple(iteration, test_time, resid / initial_b_norm);
}

// 此函数仅允许替换更快速的SpMV和Norm计算（不计入成绩），但不得改变精度
void initialize(SpM<double> *A, double *x, double *b)
{
    int N = A->nrows;

    for (int i = 0; i < N; i++)
    {
        x[i] = sin(i);
    }

    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));
    
    double* d_x;
    HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_x, x, N * sizeof(double), hipMemcpyHostToDevice));
    
    double beta;
    HIPBLAS_CHECK(hipblasDnrm2(handle, N, d_x, 1, &beta));
    
    double alpha_norm = 1.0 / beta;
    HIPBLAS_CHECK(hipblasDscal(handle, N, &alpha_norm, d_x, 1));
    
    hipsparseHandle_t sparseHandle;
    HIPSPARSE_CHECK(hipsparseCreate(&sparseHandle));
    
    uint* d_rows;
    uint* d_cols;
    double* d_vals;
    double* d_b;
    void* d_spmv_buffer_init;
    size_t bufferSize_init = 0;
    
    HIP_CHECK(hipMalloc(&d_rows, (A->nrows + 1) * sizeof(uint)));
    HIP_CHECK(hipMalloc(&d_cols, A->nnz * sizeof(uint)));
    HIP_CHECK(hipMalloc(&d_vals, A->nnz * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(double)));
    
    HIP_CHECK(hipMemcpy(d_rows, A->rows, (A->nrows + 1) * sizeof(uint), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cols, A->cols, A->nnz * sizeof(uint), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vals, A->vals, A->nnz * sizeof(double), hipMemcpyHostToDevice));
    
    hipsparseSpMatDescr_t matA_descr;
    hipsparseDnVecDescr_t vecX_descr, vecB_descr;
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA_descr, N, N, A->nnz, d_rows, d_cols, d_vals,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecX_descr, N, d_x, HIP_R_64F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecB_descr, N, d_b, HIP_R_64F));

    const double alpha_spmv = 1.0;
    const double beta_spmv = 0.0;

    HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(sparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha_spmv, matA_descr, vecX_descr, &beta_spmv, vecB_descr,
                                        HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize_init));
    HIP_CHECK(hipMalloc(&d_spmv_buffer_init, bufferSize_init));

    spmv_dcu(sparseHandle, matA_descr, vecX_descr, vecB_descr, &alpha_spmv, &beta_spmv, HIP_R_64F, d_spmv_buffer_init);
    HIP_CHECK(hipMemcpy(b, d_b, N * sizeof(double), hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_rows));
    HIP_CHECK(hipFree(d_cols));
    HIP_CHECK(hipFree(d_vals));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_spmv_buffer_init));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA_descr));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecX_descr));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecB_descr));
    HIPBLAS_CHECK(hipblasDestroy(handle));
    HIPSPARSE_CHECK(hipsparseDestroy(sparseHandle));

    for (uint i = 0; i < N; i++)
        x[i] = 0.0;
}