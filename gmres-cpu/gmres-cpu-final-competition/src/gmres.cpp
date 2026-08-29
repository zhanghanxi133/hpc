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
    const uint N = A_d->nrows;
    // 不可在此函数中修改host端A_d和_b的数据，否则会影响后续的精度评测

    //==================以下代码禁止修改=================
    double init_res = 0.0;
    for (uint i = 0; i < N; ++i)
        init_res += _b[i] * _b[i];
    init_res = std::sqrt(init_res);
    double RESID_LIMIT = REL_RESID_LIMIT * init_res;
    //==================以上代码禁止修改=================
    
    /* GMRES涉及的中间向量在CPU或DCU的内存申请和释放均可放在计时范围外，稀疏矩阵A（double类型、CSR格式）、右端向量b（double类型）、最终解x（double类型）的数据传输开销可放在计时范围外，其余所有操作均需包含在计时范围内，包括但不限于：高精度数据和低精度数据间的转换、SpMV预处理开销、稀疏矩阵预处理开销、中间计算涉及的内存拷贝开销、参数配置开销等，若要采用其他的稀疏矩阵压缩格式，需从DCU端的CSR矩阵开始转换，且格式转换的代码必须包含到计时范围内 */ 
    double alpha = 0.0;
    double beta = init_res;

    std::vector<double> r0(N);
    std::vector<double> V((RESTART_TIMES + 1) * N);
    std::vector<double> s(RESTART_TIMES + 1, 0.0);
    std::vector<double> V0(N);
    std::vector<double> H((RESTART_TIMES + 1) * RESTART_TIMES);
    std::vector<double> cs(RESTART_TIMES);
    std::vector<double> sn(RESTART_TIMES);

    int i, j, k;
    double resid;
    int iteration = 0;

    auto start = std::chrono::high_resolution_clock::now(); // 禁止修改
    // =======如果已移植为DCU代码，请更换为下列计时=====================================
    // =======如果在gmres的实现中使用了stream，需注意关联hip event计时事件和相应的stream，保证计时准确=========
    // hipEvent_t test_start_event, test_stop_event;    // 禁止修改
    // hipEventCreate(&test_start_event);               // 禁止修改
    // hipEventCreate(&test_stop_event);                // 禁止修改
    // hipEventRecord(test_start_event, 0);             // 禁止修改

    /****GMRES计算过程****/
    do
    {
        // ==========外迭代============
        spmv(A_d->rows, A_d->cols, A_d->vals, x_d, r0.data(), N);

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

    auto stop = std::chrono::high_resolution_clock::now();            // 禁止修改
    std::chrono::duration<float, std::milli> duration = stop - start; // 禁止修改
    float test_time = duration.count();                               // 禁止修改
    // =======如果已移植为DCU代码，请更换为下列计时==========
    // hipEventRecord(test_stop_event, 0);                                  //禁止修改
    // hipEventSynchronize(test_stop_event);                                //禁止修改
    // float test_time = 0.0;                                               //禁止修改
    // hipEventElapsedTime(&test_time, test_start_event, test_stop_event);  //禁止修改

    // 返回前务必将device端的最终解向量传回到host端的x_d，以用于后续的精度评测
    return make_tuple(iteration, test_time, resid / init_res);          // 禁止修改
}

