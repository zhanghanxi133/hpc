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

using namespace std;

const int RESTART_TIMES = 20;        // 禁止修改
const double REL_RESID_LIMIT = 1e-6; // 禁止修改
const int ITERATION_LIMIT = 10000;   // 禁止修改


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
    // 稀疏矩阵A_d需默认以CSR格式压缩存储，使用DCU优化时，该矩阵A和右端向量b传输到DCU端的开销，以及结果向量x传输到CPU端的开销可不算在计时范围
    // 若要采用其他的稀疏矩阵压缩格式，需从DCU端的CSR矩阵开始转换，且格式转换的代码必须包含到计时范围内
    const uint N = A_d->nrows;

    std::vector<double> r0(N);
    std::vector<double> V((RESTART_TIMES + 1) * N);
    std::vector<double> s(RESTART_TIMES + 1, 0.0);
    std::vector<double> V0(N);
    std::vector<double> H((RESTART_TIMES + 1) * RESTART_TIMES);
    std::vector<double> cs(RESTART_TIMES);
    std::vector<double> sn(RESTART_TIMES);

    double H_cpu;
    double beta_cpu;

    double alpha;
    
    double beta;
    beta = calculateNorm(_b, N);            // 禁止修改
    double RESID_LIMIT = REL_RESID_LIMIT * beta;// 禁止修改
    double init_res = beta;                     // 禁止修改

    int i, j, k;
    double resid;
    int iteration = 0;

    auto start = std::chrono::high_resolution_clock::now(); //禁止修改
    // =======如果已移植为DCU代码，请更换为下列计时==========
    // hipEvent_t test_start_event, test_stop_event;
    // hipEventCreate(&test_start_event);
    // hipEventCreate(&test_stop_event);
    // hipEventRecord(test_start_event, 0);

    // 任何对稀疏矩阵的预处理操作，如稀疏矩阵压缩格式转换、非零元数组或向量的精度转换、稀疏矩阵的非零元特征计算等，均需放在计时范围内，相关内存申请和释放除外

    /****GMRES计算过程****/
    do
    {
        // ==========外迭代============
        spmv(A_d->rows, A_d->cols, A_d->vals, x_d, r0.data(), N);  // 不可修改此步操作中相关数据的存储精度和SpMV计算精度

        alpha = -1.0;
        daxpy(alpha, _b, r0.data(), N);

        beta = calculateNorm(r0.data(), N);

        alpha = -1.0 / beta;
        dscal(alpha, r0.data(), N);

        dcopy(r0.data(), V.data(), N);

        // 初始化残差向量
        fill(s.begin(), s.end(), 0.0);
        s[0] = beta;

        resid = std::abs(beta);
        i = -1;

        if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT)
        {
            break;
        }
        do
        {
            // ==========内迭代============
            i++;
            iteration++;

            std::vector<double> V_i(N);
            dcopy(V.data() + i * N, V_i.data(), N);

            spmv(A_d->rows, A_d->cols, A_d->vals, V_i.data(), r0.data(), N);

            for (k = 0; k <= i; k++)
            {
                H[k * RESTART_TIMES + i] = dotProduct(r0.data(), V.data() + k * N, N);

                alpha = -H[k * RESTART_TIMES + i];
                daxpy(alpha, V.data() + N * k, r0.data(), N);
            }
            H[(i + 1) * RESTART_TIMES + i] = calculateNorm(r0.data(), N);

            alpha = 1.0 / H[(i + 1) * RESTART_TIMES + i];
            dscal(alpha, r0.data(), N);
            dcopy(r0.data(), V.data() + N * (i + 1), N);

            rotation2(RESTART_TIMES, H.data(), cs.data(), sn.data(), s.data(), i);

            resid = std::abs(s[i + 1]);
            // std::cout << "iteration " << iteration << ", resid = " << resid/init_res << std::endl;

            if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT)
            {
                break;
            }
        } while (i + 1 < RESTART_TIMES && iteration <= ITERATION_LIMIT);

        // 求解上三角系统
        sovlerTri(RESTART_TIMES, i, H.data(), s.data());

        // 更新解
        for (j = 0; j <= i; j++)
        {
            daxpy(s[j], V.data() + j * N, x_d, N);
        }
    } while (resid > RESID_LIMIT && iteration <= ITERATION_LIMIT);

    auto stop = std::chrono::high_resolution_clock::now();                  //禁止修改
    std::chrono::duration<float, std::milli> duration = stop - start;       //禁止修改
    float test_time = duration.count();                                     //禁止修改
    // =======如果已移植为DCU代码，请更换为下列计时==========
    // hipEventRecord(test_stop_event, 0);                                  //禁止修改  
    // hipEventSynchronize(test_stop_event);                                //禁止修改
    // float test_time = 0.0;                                               //禁止修改
    // hipEventElapsedTime(&test_time, test_start_event, test_stop_event);  //禁止修改

    return make_tuple(iteration, test_time, resid / init_res);              //禁止修改
}

// 此函数仅允许替换更快速的SpMV和Norm计算（不计入成绩），但不得改变精度
void initialize(SpM<double> *A, double *x, double *b)
{
    int N = A->nrows;

    for (int i = 0; i < N; i++)
    {
        x[i] = sin(i);
    }

    double beta = calculateNorm(x, N); // 可修改，但不可改变精度
    for (uint i = 0; i < N; i++)
    {
        x[i] /= beta;
    }

    spmv(A->rows, A->cols, A->vals, x, b, N); // 可修改，但不可改变精度

    for (uint i = 0; i < N; i++)
        x[i] = 0.0;
}
