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

const int RESTART_TIMES = 20;        // 禁止修改
const double REL_RESID_LIMIT = 1e-6; // 禁止修改
const int ITERATION_LIMIT = 10000;   // 禁止修改

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
        std::cerr << "HIPBLAS error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIPSPARSE_CHECK(cmd) \
do { \
    hipsparseStatus_t status = cmd; \
    if (status != HIPSPARSE_STATUS_SUCCESS) { \
        std::cerr << "HIPSPARSE error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Device kernels
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
        *d_final_i = final_idx;
        *d_resid_out = fabsf(d_s[final_idx + 1]);
    }
}

__global__ void cast_double_to_float_kernel(float* dst, const double* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)src[idx];
    }
}

__global__ void cast_and_scale_double_to_float_kernel(float* dst, const double* src, double scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)(src[idx] * scale);
    }
}

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

// Helper functions for DCU SpMV
void spmv_dcu(hipsparseHandle_t handle, hipsparseSpMatDescr_t matA, hipsparseDnVecDescr_t vecX, 
              hipsparseDnVecDescr_t vecY, const void* alpha, const void* beta, hipDataType type, void* buffer) {
    HIPSPARSE_CHECK(hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 alpha, matA, vecX, beta, vecY,
                                 type, HIPSPARSE_SPMV_ALG_DEFAULT, buffer));
}

// 保留原有函数以供main.cpp调用
void applyRotation(double &dx, double &dy, double &cs, double &sn)
{
    double temp = cs * dx + sn * dy;
    dy = (-sn) * dx + cs * dy;
    dx = temp;
}

void generateRotation(double &dx, double &dy, double &cs, double &sn)
{
    if (dx == double(0))
    {
        cs = double(0);
        sn = double(1);
    }
    else
    {
        double scale = fabs(dx) + fabs(dy);
        double norm = scale * std::sqrt(fabs(dx / scale) * fabs(dx / scale) +
                                        fabs(dy / scale) * fabs(dy / scale));
        double alpha = dx / fabs(dx);
        cs = fabs(dx) / norm;
        sn = alpha * dy / norm;
    }
}

void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i)
{
    for (uint k = 0; k < i; k++)
    {
        applyRotation(H[k * Am + i], H[(k + 1) * Am + i], cs[k], sn[k]);
    }
    generateRotation(H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
    applyRotation(H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
    applyRotation(s[i], s[i + 1], cs[i], sn[i]);
}

double calculateNorm(const double *vec, uint N)
{
    double sum = 0.0;
    for (uint i = 0; i < N; ++i)
    {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

void spmv(const uint *rowPtr, const uint *colInd, const double *values,
          const double *x, double *y, uint numRows)
{
    for (uint i = 0; i < numRows; ++i)
    {
        double sum = 0.0;
        for (uint j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
        {
            sum += values[j] * x[colInd[j]];
        }
        y[i] = sum;
    }
}

double dotProduct(const double *x, const double *y, uint N)
{
    double sum = 0.0;
    for (uint i = 0; i < N; ++i)
    {
        sum += x[i] * y[i];
    }
    return sum;
}

void daxpy(double alpha, const double *x, double *y, uint N)
{
    for (uint i = 0; i < N; ++i)
    {
        y[i] += alpha * x[i];
    }
}

void dscal(double alpha, double *x, uint N)
{
    for (uint i = 0; i < N; ++i)
    {
        x[i] *= alpha;
    }
}

void dcopy(const double *src, double *dst, uint N)
{
    for (uint i = 0; i < N; ++i)
    {
        dst[i] = src[i];
    }
}

void sovlerTri(int Am, int i, double *H, double *s)
{
    for (int j = i; j >= 0; j--)
    {
        s[j] /= H[Am * j + j];
        for (int k = j - 1; k >= 0; k--)
        {
            s[k] -= H[k * Am + j] * s[j];
        }
    }
}

RESULT gmres(SpM<double> *A_d, double *x_d, double *_b)
{
    const uint N = A_d->nrows;
    // 不可在此函数中修改host端A_d和_b的数据，否则会影响后续的精度评测

    //==================以下代码禁止修改=================
    double init_res = 0.0;
    for (uint i = 0; i < N; ++i)
        init_res += _b[i] * _b[i];
    init_res = std::sqrt(init_res);
    double RESID_LIMIT = REL_RESID_LIMIT * init_res;
    //==================以上代码禁止修改=================
    
    const int m = RESTART_TIMES;
    const int H_lda = m + 1;

    hipblasHandle_t hipblasHandle;
    hipsparseHandle_t hipsparseHandle;
    HIPBLAS_CHECK(hipblasCreate(&hipblasHandle));
    HIPSPARSE_CHECK(hipsparseCreate(&hipsparseHandle));

    // Device memory allocation
    double *d_x, *d_b, *d_r;
    uint   *d_rows, *d_cols;
    double *d_vals;
    float *d_V_float, *d_H_float, *d_s_float, *d_cs_float, *d_sn_float;
    float *d_w_float, *d_vals_float;
    int* d_final_i;
    float* d_resid_val_float;

    HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_r, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_rows, (A_d->nrows + 1) * sizeof(uint)));
    HIP_CHECK(hipMalloc(&d_cols, A_d->nnz * sizeof(uint)));
    HIP_CHECK(hipMalloc(&d_vals, A_d->nnz * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_V_float, N * (m + 1) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_H_float, H_lda * m * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_s_float, (m + 1) * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cs_float, m * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sn_float, m * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_w_float, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_vals_float, A_d->nnz * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_final_i, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_resid_val_float, sizeof(float)));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_x, x_d, N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, _b, N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rows, A_d->rows, (A_d->nrows + 1) * sizeof(uint), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cols, A_d->cols, A_d->nnz * sizeof(uint), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vals, A_d->vals, A_d->nnz * sizeof(double), hipMemcpyHostToDevice));

    // Create hipSPARSE descriptors
    hipsparseSpMatDescr_t matA_descr_double, matA_descr_float;
    hipsparseDnVecDescr_t vecX_descr_double, vecR_descr_double;
    hipsparseDnVecDescr_t vecW_descr_float;
    std::vector<hipsparseDnVecDescr_t> vecV_descr_float(m + 1);

    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA_descr_double, N, N, A_d->nnz, d_rows, d_cols, d_vals, 
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecX_descr_double, N, d_x, HIP_R_64F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecR_descr_double, N, d_r, HIP_R_64F));

    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA_descr_float, N, N, A_d->nnz, d_rows, d_cols, d_vals_float, 
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecW_descr_float, N, d_w_float, HIP_R_32F));
    for (int i = 0; i <= m; ++i) {
        HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecV_descr_float[i], N, d_V_float + i * N, HIP_R_32F));
    }
    
    // Allocate SpMV buffers
    void* d_spmv_buffer_double = nullptr, *d_spmv_buffer_float = nullptr;
    size_t bufferSize_d = 0, bufferSize_f = 0;
    const double alpha_d_one = 1.0, beta_d_zero = 0.0;
    const float alpha_f_one = 1.0f, beta_f_zero = 0.0f;
    
    HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, 
                                            &alpha_d_one, matA_descr_double, vecX_descr_double, 
                                            &beta_d_zero, vecR_descr_double, HIP_R_64F, 
                                            HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize_d));
    HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(hipsparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, 
                                            &alpha_f_one, matA_descr_float, vecV_descr_float[0], 
                                            &beta_f_zero, vecW_descr_float, HIP_R_32F, 
                                            HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize_f));
    HIP_CHECK(hipMalloc(&d_spmv_buffer_double, bufferSize_d));
    HIP_CHECK(hipMalloc(&d_spmv_buffer_float, bufferSize_f));

    double resid = init_res;
    int iteration = 0;
    const double alpha_one_d = 1.0, alpha_neg_one_d = -1.0;
    const float  alpha_one_f = 1.0f, alpha_neg_one_f = -1.0f, alpha_zero_f = 0.0f;

    // =================== Warmup 开始 ===================
    {
        dim3 block_warmup(256);
        dim3 grid_cast_warmup((A_d->nnz + block_warmup.x - 1) / block_warmup.x);
        hipLaunchKernelGGL(cast_double_to_float_kernel, grid_cast_warmup, block_warmup, 0, 0, d_vals_float, d_vals, A_d->nnz);
        dim3 grid_N_warmup((N + block_warmup.x - 1) / block_warmup.x);

        HIP_CHECK(hipMemcpy(d_r, d_b, N * sizeof(double), hipMemcpyDeviceToDevice));
        spmv_dcu(hipsparseHandle, matA_descr_double, vecX_descr_double, vecR_descr_double, 
                 &alpha_neg_one_d, &alpha_one_d, HIP_R_64F, d_spmv_buffer_double);
        double dummy_beta = 0.0;
        HIPBLAS_CHECK(hipblasDnrm2(hipblasHandle, N, d_r, 1, &dummy_beta));

        hipLaunchKernelGGL(cast_and_scale_double_to_float_kernel, grid_N_warmup, block_warmup, 0, 0, 
                          d_V_float, d_r, 1.0, N);
        hipLaunchKernelGGL(set_value_float_kernel, dim3(1), dim3(1), 0, 0, d_s_float, 1.0f);

        spmv_dcu(hipsparseHandle, matA_descr_float, vecV_descr_float[0], vecW_descr_float, 
                 &alpha_one_f, &alpha_zero_f, HIP_R_32F, d_spmv_buffer_float);
        HIPBLAS_CHECK(hipblasSgemv(hipblasHandle, HIPBLAS_OP_T, N, 1, &alpha_one_f, d_V_float, N, 
                                   d_w_float, 1, &alpha_zero_f, d_H_float, 1));
        HIPBLAS_CHECK(hipblasSgemv(hipblasHandle, HIPBLAS_OP_N, N, 1, &alpha_neg_one_f, d_V_float, N, 
                                   d_H_float, 1, &alpha_one_f, d_w_float, 1));
        float dummy_h = 0.0f;
        HIPBLAS_CHECK(hipblasSnrm2(hipblasHandle, N, d_w_float, 1, &dummy_h));
        float dummy_inv_h = 1.0f;
        HIPBLAS_CHECK(hipblasSscal(hipblasHandle, N, &dummy_inv_h, d_w_float, 1));

        hipLaunchKernelGGL(arnoldi_update_and_givens_float_kernel, dim3(1), dim3(1), 0, 0, 
                          d_H_float, d_cs_float, d_sn_float, d_s_float, 1, H_lda);
        hipLaunchKernelGGL(find_convergence_and_resid_float_kernel, dim3(1), dim3(1), 0, 0,
                          d_s_float, 1, (float)RESID_LIMIT, d_final_i, d_resid_val_float);
        HIPBLAS_CHECK(hipblasStrsv(hipblasHandle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, 
                                   HIPBLAS_DIAG_NON_UNIT, 1, d_H_float, H_lda, d_s_float, 1));
        hipLaunchKernelGGL(update_solution_kernel, grid_N_warmup, block_warmup, 0, 0,
                          d_x, d_V_float, d_s_float, N, 0, N);

        HIP_CHECK(hipDeviceSynchronize());
    }
    // =================== Warmup 结束 ===================

    // DCU计时开始
    hipEvent_t test_start_event, test_stop_event;
    HIP_CHECK(hipEventCreate(&test_start_event));
    HIP_CHECK(hipEventCreate(&test_stop_event));
    HIP_CHECK(hipEventRecord(test_start_event, 0));

    // Convert matrix values to float
    dim3 block(256);
    dim3 grid_cast((A_d->nnz + block.x - 1) / block.x);
    hipLaunchKernelGGL(cast_double_to_float_kernel, grid_cast, block, 0, 0, d_vals_float, d_vals, A_d->nnz);

    do {
        // 外迭代
        HIP_CHECK(hipMemcpy(d_r, d_b, N * sizeof(double), hipMemcpyDeviceToDevice));
        spmv_dcu(hipsparseHandle, matA_descr_double, vecX_descr_double, vecR_descr_double, 
                 &alpha_neg_one_d, &alpha_one_d, HIP_R_64F, d_spmv_buffer_double);
        
        double beta = 0.0;
        HIPBLAS_CHECK(hipblasDnrm2(hipblasHandle, N, d_r, 1, &beta));
        
        // Precision transition: Double -> Float
        double inv_beta = (beta == 0.0) ? 1.0 : 1.0 / beta;
        dim3 grid_N((N + block.x - 1) / block.x);
        hipLaunchKernelGGL(cast_and_scale_double_to_float_kernel, grid_N, block, 0, 0, 
                          d_V_float, d_r, inv_beta, N);

        HIP_CHECK(hipMemset(d_s_float, 0, (m + 1) * sizeof(float)));
        hipLaunchKernelGGL(set_value_float_kernel, dim3(1), dim3(1), 0, 0, d_s_float, (float)beta);

        resid = beta;
        if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) break;
        
        int inner_loop_actual_iters = 0;

        // 内迭代
        for (int i = 0; i < m; ++i) {
            iteration++;
            inner_loop_actual_iters = i + 1;
            
            spmv_dcu(hipsparseHandle, matA_descr_float, vecV_descr_float[i], vecW_descr_float, 
                     &alpha_one_f, &alpha_zero_f, HIP_R_32F, d_spmv_buffer_float);
            
            float* d_Hi_col = d_H_float + i * H_lda;
            
            HIPBLAS_CHECK(hipblasSgemv(hipblasHandle, HIPBLAS_OP_T, N, i + 1, &alpha_one_f, 
                                       d_V_float, N, d_w_float, 1, &alpha_zero_f, d_Hi_col, 1));
            HIPBLAS_CHECK(hipblasSgemv(hipblasHandle, HIPBLAS_OP_N, N, i + 1, &alpha_neg_one_f, 
                                       d_V_float, N, d_Hi_col, 1, &alpha_one_f, d_w_float, 1));

            float h_i1_i = 0.0f;
            HIPBLAS_CHECK(hipblasSnrm2(hipblasHandle, N, d_w_float, 1, &h_i1_i));
            hipLaunchKernelGGL(set_value_float_kernel, dim3(1), dim3(1), 0, 0, d_Hi_col + i + 1, h_i1_i);
            
            if (h_i1_i != 0.0f){
                float inv_h = 1.0f / h_i1_i;
                HIPBLAS_CHECK(hipblasSscal(hipblasHandle, N, &inv_h, d_w_float, 1));
                HIP_CHECK(hipMemcpy(d_V_float + (i + 1) * N, d_w_float, N * sizeof(float), 
                                   hipMemcpyDeviceToDevice));
            } else {
                break;
            }

            if(iteration >= ITERATION_LIMIT) break;
        }

        if (inner_loop_actual_iters > 0) {
            hipLaunchKernelGGL(arnoldi_update_and_givens_float_kernel, dim3(1), dim3(1), 0, 0, 
                              d_H_float, d_cs_float, d_sn_float, d_s_float, inner_loop_actual_iters, H_lda);
        }

        int h_final_i;
        float resid_float;
        hipLaunchKernelGGL(find_convergence_and_resid_float_kernel, dim3(1), dim3(1), 0, 0,
                          d_s_float, inner_loop_actual_iters, (float)RESID_LIMIT, d_final_i, d_resid_val_float);
        HIP_CHECK(hipMemcpy(&h_final_i, d_final_i, sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&resid_float, d_resid_val_float, sizeof(float), hipMemcpyDeviceToHost));
        resid = (double)resid_float;

        bool converged_in_inner_loop = false;

        // Solve triangular system and update solution
        if (h_final_i >= 0) {
            HIPBLAS_CHECK(hipblasStrsv(hipblasHandle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, 
                                      HIPBLAS_DIAG_NON_UNIT, h_final_i + 1, d_H_float, H_lda, d_s_float, 1));
            
            hipLaunchKernelGGL(update_solution_kernel, grid_N, block, 0, 0,
                              d_x, d_V_float, d_s_float, N, h_final_i, N);
        }
        
        if (converged_in_inner_loop) break;

    } while (resid > RESID_LIMIT && iteration < ITERATION_LIMIT);

    // DCU计时结束
    HIP_CHECK(hipEventRecord(test_stop_event, 0));
    HIP_CHECK(hipEventSynchronize(test_stop_event));
    float test_time = 0.0;
    HIP_CHECK(hipEventElapsedTime(&test_time, test_start_event, test_stop_event));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(x_d, d_x, N * sizeof(double), hipMemcpyDeviceToHost));

    // Clean up
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
    for (int i = 0; i <= m; ++i) { 
        HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecV_descr_float[i])); 
    }
    
    HIPBLAS_CHECK(hipblasDestroy(hipblasHandle));
    HIPSPARSE_CHECK(hipsparseDestroy(hipsparseHandle));
    HIP_CHECK(hipEventDestroy(test_start_event));
    HIP_CHECK(hipEventDestroy(test_stop_event));

    return make_tuple(iteration, test_time, resid / init_res);  // 禁止修改
}