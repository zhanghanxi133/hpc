#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cfloat>
#include "sparseMatrix.hpp"
#include "gmres.hpp"
#include <hip/hip_runtime.h>

using namespace std;

const int RESTART_TIMES = 20;
const double REL_RESID_LIMIT = 1e-6;
const int ITERATION_LIMIT = 10000;

constexpr int TPB = 256;
constexpr int MAX_M = RESTART_TIMES + 1;

#if defined(__HIP_PLATFORM_AMD__)
constexpr int WARP = 64;
#else
constexpr int WARP = 32;
#endif

#define HIP_CHECK(cmd) \
do { \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

inline int grid_for_vec(int n) { return (n + TPB - 1) / TPB; }
inline int grid_for_dot(int n) { return std::max(1, std::min((n + 8*TPB - 1)/(8*TPB), 4096)); }

// ============ CPU端函数（main.cpp调用） ============
void applyRotation(double &dx, double &dy, double &cs, double &sn) {
    double temp = cs * dx + sn * dy;
    dy = (-sn) * dx + cs * dy;
    dx = temp;
}

void generateRotation(double &dx, double &dy, double &cs, double &sn) {
    if (dx == double(0)) {
        cs = double(0);
        sn = double(1);
    } else {
        double scale = fabs(dx) + fabs(dy);
        double norm = scale * std::sqrt(fabs(dx/scale)*fabs(dx/scale) + fabs(dy/scale)*fabs(dy/scale));
        double alpha = dx / fabs(dx);
        cs = fabs(dx) / norm;
        sn = alpha * dy / norm;
    }
}

void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i) {
    for (uint k = 0; k < i; k++) {
        applyRotation(H[k * Am + i], H[(k + 1) * Am + i], cs[k], sn[k]);
    }
    generateRotation(H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
    applyRotation(H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
    applyRotation(s[i], s[i + 1], cs[i], sn[i]);
}

double calculateNorm(const double *vec, uint N) {
    double sum = 0.0;
    for (uint i = 0; i < N; ++i)
        sum += vec[i] * vec[i];
    return std::sqrt(sum);
}

void spmv(const uint *rowPtr, const uint *colInd, const double *values,
          const double *x, double *y, uint numRows) {
    for (uint i = 0; i < numRows; ++i) {
        double sum = 0.0;
        for (uint j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
            sum += values[j] * x[colInd[j]];
        }
        y[i] = sum;
    }
}

double dotProduct(const double *x, const double *y, uint N) {
    double sum = 0.0;
    for (uint i = 0; i < N; ++i)
        sum += x[i] * y[i];
    return sum;
}

void daxpy(double alpha, const double *x, double *y, uint N) {
    for (uint i = 0; i < N; ++i)
        y[i] += alpha * x[i];
}

void dscal(double alpha, double *x, uint N) {
    for (uint i = 0; i < N; ++i)
        x[i] *= alpha;
}

void dcopy(const double *src, double *dst, uint N) {
    for (uint i = 0; i < N; ++i)
        dst[i] = src[i];
}

void sovlerTri(int Am, int i, double *H, double *s) {
    for (int j = i; j >= 0; j--) {
        s[j] /= H[Am * j + j];
        const double sj = s[j];
        for (int k = j - 1; k >= 0; k--) {
            s[k] -= H[k * Am + j] * sj;
        }
    }
}

// ============ Device Kernels ============
__device__ inline double warp_reduce_sum(double v) {
    for (int off = (WARP >> 1); off > 0; off >>= 1)
        v += __shfl_down(v, off, WARP);
    return v;
}

__global__ void cast_d2f(int n, const double *__restrict__ in, float *__restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}

__global__ void cast_and_scale_d2f(int n, const double *__restrict__ in, double scale, float *__restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)(in[i] * scale);
}

__global__ void scale_and_copy_f32(int n, const float *__restrict__ in, float scale, float *__restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * scale;
}

__global__ void vec_axpy(int n, double alpha, const double *__restrict__ x, double *__restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

// 点积 partial
__global__ void dot_partial_kernel(int n, const double *__restrict__ x, const double *__restrict__ y,
                                   double *__restrict__ partial) {
    __shared__ double sdata[TPB];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    
    double sum = 0.0;
    for (int i = idx; i < n; i += stride) {
        sum += x[i] * y[i];
    }
    
    sum = warp_reduce_sum(sum);
    __shared__ double warp_sums[TPB/WARP];
    int lane = threadIdx.x & (WARP - 1);
    int wid = threadIdx.x / WARP;
    
    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();
    
    if (wid == 0) {
        double block_sum = (lane < (TPB/WARP)) ? warp_sums[lane] : 0.0;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) partial[blockIdx.x] = block_sum;
    }
}

__global__ void reduce_sum_kernel(const double *__restrict__ in, int n, double *__restrict__ out) {
    __shared__ double sdata[TPB];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int i = tid; i < n; i += blockDim.x)
        sum += in[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = TPB/2; s >= 1; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = sdata[0];
}

// 批量点积（float读，double累加）
__global__ void dot_multi_partial_f32acc_f64(
    int n, int m, const float *__restrict__ w, const float *__restrict__ V, int ldv,
    double *__restrict__ partial) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    double acc[MAX_M];
    #pragma unroll
    for (int j = 0; j < MAX_M; ++j)
        if (j < m) acc[j] = 0.0;
    
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < n; i += stride) {
        double wi = (double)w[i];
        #pragma unroll
        for (int j = 0; j < MAX_M; ++j) {
            if (j < m) acc[j] += wi * (double)V[i + j * ldv];
        }
    }
    
    __shared__ double sdata[TPB];
    for (int j = 0; j < m; ++j) {
        sdata[tid] = acc[j];
        __syncthreads();
        for (int s = TPB/2; s >= 1; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) partial[blockIdx.x * m + j] = sdata[0];
        __syncthreads();
    }
}

__global__ void reduce_multi_kernel(const double *__restrict__ partial, int m, int nblocks,
                                    double *__restrict__ out) {
    __shared__ double sdata[TPB];
    int tid = threadIdx.x;
    for (int j = 0; j < m; ++j) {
        double sum = 0.0;
        for (int i = tid; i < nblocks; i += blockDim.x)
            sum += partial[i * m + j];
        sdata[tid] = sum;
        __syncthreads();
        for (int s = TPB/2; s >= 1; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) out[j] = sdata[0];
        __syncthreads();
    }
}

// subproject + norm
__global__ void subproject_and_partial_f32acc_f64(
    int n, int m, const float *__restrict__ V, int ldv,
    const double *__restrict__ h, float *__restrict__ r,
    double *__restrict__ partial_vec) {
    __shared__ double sdata[TPB];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    double r_i2 = 0.0;
    if (i < n) {
        double proj_sum = 0.0;
        for (int j = 0; j < m; ++j) {
            proj_sum += (double)V[i + j * ldv] * h[j];
        }
        float ri = r[i] - (float)proj_sum;
        r[i] = ri;
        r_i2 = (double)ri * (double)ri;
    }
    
    sdata[tid] = r_i2;
    __syncthreads();
    for (int s = TPB/2; s >= 1; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_vec[blockIdx.x] = sdata[0];
}

// x += V*y (float V, double y)
__global__ void add_linear_comb_mixed(int n, int m, const float *__restrict__ V, int ldv,
                                      const double *__restrict__ coeff, double *__restrict__ x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double acc = 0.0;
    for (int j = 0; j < m; ++j) {
        acc += (double)V[i + j * ldv] * coeff[j];
    }
    x[i] += acc;
}

// SpMV - 简化版（vector kernel，双精度外层，融合-b）
template<typename IndexType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__global__ void spmv_vector_minus_b_partial(
    const IndexType row_num, const IndexType *__restrict__ rowPtr,
    const IndexType *__restrict__ colInd, const double *__restrict__ vals,
    const double *__restrict__ x, const double *__restrict__ b,
    double *__restrict__ r, double *__restrict__ partial) {
    
    const IndexType tid = VECTORS_PER_BLOCK * THREADS_PER_VECTOR * blockIdx.x + threadIdx.x;
    const IndexType lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    const IndexType v_id = threadIdx.x / THREADS_PER_VECTOR;
    const IndexType row = tid / THREADS_PER_VECTOR;
    
    __shared__ double s_r2[VECTORS_PER_BLOCK];
    if (lane == 0) s_r2[v_id] = 0.0;
    
    if (row < row_num) {
        IndexType row_start = rowPtr[row];
        IndexType row_end = rowPtr[row + 1];
        
        double br = 0.0;
        if (lane == 0) br = b[row];
        
        double sum = 0.0;
        for (IndexType jj = row_start + lane; jj < row_end; jj += THREADS_PER_VECTOR) {
            sum += vals[jj] * x[colInd[jj]];
        }
        
        for (unsigned off = THREADS_PER_VECTOR >> 1; off > 0; off >>= 1)
            sum += __shfl_down(sum, off, THREADS_PER_VECTOR);
        
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
        for (unsigned v = 0; v < VECTORS_PER_BLOCK; ++v)
            blk += s_r2[v];
        partial[blockIdx.x] = blk;
    }
}

// SpMV - float版本
template<typename IndexType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__global__ void spmv_vector_f32(
    const IndexType row_num, const IndexType *__restrict__ rowPtr,
    const IndexType *__restrict__ colInd, const float *__restrict__ vals,
    const float *__restrict__ x, float *__restrict__ y) {
    
    const IndexType tid = VECTORS_PER_BLOCK * THREADS_PER_VECTOR * blockIdx.x + threadIdx.x;
    const IndexType lane = threadIdx.x & (THREADS_PER_VECTOR - 1);
    const IndexType row = tid / THREADS_PER_VECTOR;
    
    if (row >= row_num) return;
    
    IndexType row_start = rowPtr[row];
    IndexType row_end = rowPtr[row + 1];
    
    float sum = 0.0f;
    for (IndexType jj = row_start + lane; jj < row_end; jj += THREADS_PER_VECTOR) {
        sum += vals[jj] * x[colInd[jj]];
    }
    
    for (unsigned off = THREADS_PER_VECTOR >> 1; off > 0; off >>= 1)
        sum += __shfl_down(sum, off, THREADS_PER_VECTOR);
    
    if (lane == 0) y[row] = sum;
}

// ============ Helper Functions ============
static inline double device_dot(int n, const double *d_x, const double *d_y,
                                double *d_partial, double *d_out) {
    int grid = grid_for_dot(n);
    dot_partial_kernel<<<grid, TPB>>>(n, d_x, d_y, d_partial);
    reduce_sum_kernel<<<1, TPB>>>(d_partial, grid, d_out);
    double h = 0.0;
    hipMemcpy(&h, d_out, sizeof(double), hipMemcpyDeviceToHost);
    return h;
}

static inline double device_nrm2(int n, const double *d_x, double *d_partial, double *d_out) {
    double ss = device_dot(n, d_x, d_x, d_partial, d_out);
    return std::sqrt(ss);
}

static inline void device_multi_dot_f32acc_f64(int n, int m, const float *d_wf, const float *d_Vf, int ldv,
                                               double *d_partial_multi, double *d_out_multi) {
    int grid = grid_for_dot(n);
    dot_multi_partial_f32acc_f64<<<grid, TPB>>>(n, m, d_wf, d_Vf, ldv, d_partial_multi);
    reduce_multi_kernel<<<1, TPB>>>(d_partial_multi, m, grid, d_out_multi);
}

// ============ GMRES主函数 ============
RESULT gmres(SpM<double> *A_d, double *x_d, double *_b) {
    const uint N = A_d->nrows;
    const uint nnz = A_d->rows[N];
    
    //==================以下代码禁止修改=================
    double init_res = 0.0;
    for (uint i = 0; i < N; ++i)
        init_res += _b[i] * _b[i];
    init_res = std::sqrt(init_res);
    double RESID_LIMIT = REL_RESID_LIMIT * init_res;
    //==================以上代码禁止修改=================
    
    const int m = RESTART_TIMES;
    
    // Host端数据
    std::vector<double> H((m + 1) * m, 0.0);
    std::vector<double> cs(m, 0.0), sn(m, 0.0);
    std::vector<double> s(m + 1, 0.0);
    std::vector<double> h_host(MAX_M, 0.0);
    
    // Device内存分配
    unsigned *d_rowPtr = nullptr, *d_colInd = nullptr;
    double *d_vals = nullptr, *d_x = nullptr, *d_b = nullptr, *d_r = nullptr;
    float *d_vals_f = nullptr, *d_Vf = nullptr, *d_rf = nullptr;
    double *d_partial = nullptr, *d_dotOut = nullptr;
    double *d_partial_multi = nullptr, *d_h = nullptr, *d_y = nullptr;
    double *d_partial_vec = nullptr, *d_partial_spmv = nullptr;
    
    HIP_CHECK(hipMalloc(&d_rowPtr, (N + 1) * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&d_colInd, nnz * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&d_vals, nnz * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_vals_f, nnz * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_r, N * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Vf, (m + 1) * (size_t)N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rf, N * sizeof(float)));
    
    int maxBlocks = grid_for_dot(N);
    HIP_CHECK(hipMalloc(&d_partial, maxBlocks * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_dotOut, sizeof(double)));
    HIP_CHECK(hipMalloc(&d_partial_multi, maxBlocks * MAX_M * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_h, MAX_M * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y, m * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_partial_vec, grid_for_vec(N) * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_partial_spmv, ((N + 4 - 1) / 4) * sizeof(double)));
    
    // H2D拷贝
    HIP_CHECK(hipMemcpy(d_rowPtr, A_d->rows, (N + 1) * sizeof(unsigned), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_colInd, A_d->cols, nnz * sizeof(unsigned), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vals, A_d->vals, nnz * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, x_d, N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, _b, N * sizeof(double), hipMemcpyHostToDevice));
    
    // =================== Warmup 开始 ===================
    {
        cast_d2f<<<grid_for_vec(nnz), TPB>>>(nnz, d_vals, d_vals_f);
        constexpr unsigned V = 4, TV = 16;
        dim3 block(V * TV), grid((N + V - 1) / V);
        spmv_vector_minus_b_partial<uint, V, TV><<<grid, block>>>(N, d_rowPtr, d_colInd, d_vals, d_x, d_b, d_r, d_partial_spmv);
        spmv_vector_f32<uint, V, TV><<<grid, block>>>(N, d_rowPtr, d_colInd, d_vals_f, d_Vf, d_rf);
        HIP_CHECK(hipDeviceSynchronize());
    }
    // =================== Warmup 结束 ===================
    
    // 计时开始
    hipEvent_t test_start_event, test_stop_event;
    HIP_CHECK(hipEventCreate(&test_start_event));
    HIP_CHECK(hipEventCreate(&test_stop_event));
    HIP_CHECK(hipEventRecord(test_start_event, 0));
    
    // 转换矩阵到float
    cast_d2f<<<grid_for_vec(nnz), TPB>>>(nnz, d_vals, d_vals_f);
    
    uint avg_nnz_per_row = std::max(1u, nnz / N);
    unsigned V = 4, TV = 16;
    if (avg_nnz_per_row <= 16) TV = 4;
    else if (avg_nnz_per_row <= 32) TV = 8;
    else if (avg_nnz_per_row <= 64) TV = 16;
    else TV = 32;
    
    int iteration = 0;
    double resid = 0.0;
    
    // ====== GMRES 主循环 ======
    do {
        // 外层残差：r = A*x - b, beta = ||r||
        double ss_beta = 0.0;
        dim3 block_outer(V * TV), grid_outer((N + V - 1) / V);
        
        if (TV == 4) {
            spmv_vector_minus_b_partial<uint, 4, 4><<<grid_outer, block_outer>>>(
                N, d_rowPtr, d_colInd, d_vals, d_x, d_b, d_r, d_partial_spmv);
        } else if (TV == 8) {
            spmv_vector_minus_b_partial<uint, 4, 8><<<grid_outer, block_outer>>>(
                N, d_rowPtr, d_colInd, d_vals, d_x, d_b, d_r, d_partial_spmv);
        } else if (TV == 16) {
            spmv_vector_minus_b_partial<uint, 4, 16><<<grid_outer, block_outer>>>(
                N, d_rowPtr, d_colInd, d_vals, d_x, d_b, d_r, d_partial_spmv);
        } else {
            spmv_vector_minus_b_partial<uint, 4, 32><<<grid_outer, block_outer>>>(
                N, d_rowPtr, d_colInd, d_vals, d_x, d_b, d_r, d_partial_spmv);
        }
        
        reduce_sum_kernel<<<1, TPB>>>(d_partial_spmv, (N + V - 1) / V, d_dotOut);
        HIP_CHECK(hipMemcpy(&ss_beta, d_dotOut, sizeof(double), hipMemcpyDeviceToHost));
        
        // 数值安全
        if (!std::isfinite(ss_beta) || ss_beta < 0.0) ss_beta = 0.0;
        double beta = std::sqrt(ss_beta);
        
        // V[0] = -r / beta (转float)
        cast_and_scale_d2f<<<grid_for_vec(N), TPB>>>(N, d_r, (beta == 0.0 ? 0.0 : -1.0/beta), d_Vf + 0);
        
        // 初始化s
        std::fill(s.begin(), s.end(), 0.0);
        s[0] = beta;
        resid = fabs(beta);
        
        if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) break;
        
        int i = -1;
        do {
            i++;
            iteration++;
            
            // w = A * V[i] (float)
            if (TV == 4) {
                spmv_vector_f32<uint, 4, 4><<<grid_outer, block_outer>>>(
                    N, d_rowPtr, d_colInd, d_vals_f, d_Vf + (size_t)i * N, d_rf);
            } else if (TV == 8) {
                spmv_vector_f32<uint, 4, 8><<<grid_outer, block_outer>>>(
                    N, d_rowPtr, d_colInd, d_vals_f, d_Vf + (size_t)i * N, d_rf);
            } else if (TV == 16) {
                spmv_vector_f32<uint, 4, 16><<<grid_outer, block_outer>>>(
                    N, d_rowPtr, d_colInd, d_vals_f, d_Vf + (size_t)i * N, d_rf);
            } else {
                spmv_vector_f32<uint, 4, 32><<<grid_outer, block_outer>>>(
                    N, d_rowPtr, d_colInd, d_vals_f, d_Vf + (size_t)i * N, d_rf);
            }
            
            // h = V^T * w (批量点积，float读double累加)
            device_multi_dot_f32acc_f64(N, i + 1, d_rf, d_Vf, N, d_partial_multi, d_h);
            HIP_CHECK(hipMemcpy(h_host.data(), d_h, (i + 1) * sizeof(double), hipMemcpyDeviceToHost));
            
            for (int k = 0; k <= i; ++k) {
                H[k * m + i] = h_host[k];
            }
            
            // r = w - V*h, ||r||
            subproject_and_partial_f32acc_f64<<<grid_for_vec(N), TPB>>>(
                N, i + 1, d_Vf, N, d_h, d_rf, d_partial_vec);
            reduce_sum_kernel<<<1, TPB>>>(d_partial_vec, grid_for_vec(N), d_dotOut);
            
            double ss = 0.0;
            HIP_CHECK(hipMemcpy(&ss, d_dotOut, sizeof(double), hipMemcpyDeviceToHost));
            if (!std::isfinite(ss) || ss < 0.0) ss = 0.0;
            double h_ip1_i = std::sqrt(ss);
            H[(i + 1) * m + i] = h_ip1_i;
            
            // V[i+1] = r / h_{i+1,i}
            if (h_ip1_i != 0.0) {
                float scl = (float)(1.0 / h_ip1_i);
                scale_and_copy_f32<<<grid_for_vec(N), TPB>>>(N, d_rf, scl, d_Vf + (size_t)(i + 1) * N);
            } else {
                HIP_CHECK(hipMemcpy(d_Vf + (size_t)(i + 1) * N, d_rf, N * sizeof(float), hipMemcpyDeviceToDevice));
            }
            
            // CPU端Givens旋转
            rotation2(m, H.data(), cs.data(), sn.data(), s.data(), i);
            resid = fabs(s[i + 1]);
            
            // 不再提前因resid满足阈值而退出内迭代，确保每次restart执行满m步（除非达到ITERATION_LIMIT）
            if (iteration >= ITERATION_LIMIT) break;
            
        } while (i + 1 < m && iteration < ITERATION_LIMIT);
        
        // 回代
        sovlerTri(m, i, H.data(), s.data());
        
        // x += V * s
        HIP_CHECK(hipMemcpy(d_y, s.data(), (i + 1) * sizeof(double), hipMemcpyHostToDevice));
        add_linear_comb_mixed<<<grid_for_vec(N), TPB>>>(N, i + 1, d_Vf, N, d_y, d_x);
        
    } while (resid > RESID_LIMIT && iteration <= ITERATION_LIMIT);
    
    // 计时结束
    HIP_CHECK(hipEventRecord(test_stop_event, 0));
    HIP_CHECK(hipEventSynchronize(test_stop_event));
    float test_time = 0.0;
    HIP_CHECK(hipEventElapsedTime(&test_time, test_start_event, test_stop_event));
    
    // 拷贝结果
    HIP_CHECK(hipMemcpy(x_d, d_x, N * sizeof(double), hipMemcpyDeviceToHost));
    
    // 清理
    HIP_CHECK(hipFree(d_rowPtr));
    HIP_CHECK(hipFree(d_colInd));
    HIP_CHECK(hipFree(d_vals));
    HIP_CHECK(hipFree(d_vals_f));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_r));
    HIP_CHECK(hipFree(d_Vf));
    HIP_CHECK(hipFree(d_rf));
    HIP_CHECK(hipFree(d_partial));
    HIP_CHECK(hipFree(d_dotOut));
    HIP_CHECK(hipFree(d_partial_multi));
    HIP_CHECK(hipFree(d_h));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_partial_vec));
    HIP_CHECK(hipFree(d_partial_spmv));
    HIP_CHECK(hipEventDestroy(test_start_event));
    HIP_CHECK(hipEventDestroy(test_stop_event));
    
    return make_tuple(iteration, test_time, resid / init_res);
}