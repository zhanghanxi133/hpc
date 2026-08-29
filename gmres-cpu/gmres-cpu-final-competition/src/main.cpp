#include <iostream>
#include <vector>
#include <fstream>
#include "sparseMatrix.hpp"
#include "gmres.hpp"
#include "hipblas.h"
#include "hipsparse.h"

//========================================
//              此文件禁止修改
//========================================


using RESULT=std::tuple<int, float, double>;

void loadBinToCSR(SpM<double> &A, std::string mtx_path)
{
    std::ifstream binFile(mtx_path, std::ios::in | std::ios::binary);
    
    if (!binFile.is_open()) {
        std::cerr << "can not open output file!" << std::endl;
        return;
    }
    binFile.read(reinterpret_cast<char*>(&A.nrows), sizeof(A.nrows));
    binFile.read(reinterpret_cast<char*>(&A.ncols), sizeof(A.ncols));
    binFile.read(reinterpret_cast<char*>(&A.nnz), sizeof(A.nnz));

    A.rows = (uint *)malloc((A.nrows + 1) * sizeof(uint));
    A.cols = (uint *)malloc(A.nnz * sizeof(uint));
    A.vals = (double *)malloc(A.nnz * sizeof(double));

    binFile.read(reinterpret_cast<char*>(A.rows), (A.nrows + 1) * sizeof(uint));
    binFile.read(reinterpret_cast<char*>(A.cols), A.nnz * sizeof(uint));
    binFile.read(reinterpret_cast<char*>(A.vals), A.nnz * sizeof(double));
    binFile.close();
}

void initialize(SpM<double> *A, double *x, double *b)
{
    int N = A->nrows;

    for (int i = 0; i < N; i++)
    {
        x[i] = sin(i);
    }

    double beta = calculateNorm(x, N);  // 不可修改该函数的实现
    for (uint i = 0; i < N; i++)
    {
        x[i] /= beta;
    }

    spmv(A->rows, A->cols, A->vals, x, b, N); // 不可修改该函数的实现

    for (uint i = 0; i < N; i++)
        x[i] = 0.0;
}

double validate_result(SpM<double> *A_d, double *x, double *b)
{
    // 计算初始残差
    double init_res = 0.0;
    for (uint i = 0; i < A_d->nrows; ++i)
        init_res += b[i] * b[i];
    init_res = std::sqrt(init_res);

    // 创建矩阵
    uint *rows_dev_eval, *cols_dev_eval;
    double *vals_dev_eval;
    hipMalloc((void **)&rows_dev_eval, (A_d->nrows + 1) * sizeof(uint));
    hipMalloc((void **)&cols_dev_eval, A_d->nnz * sizeof(uint));
    hipMalloc((void **)&vals_dev_eval, A_d->nnz * sizeof(double));
    hipMemcpy(rows_dev_eval, A_d->rows, (A_d->nrows + 1) * sizeof(uint), hipMemcpyHostToDevice);
    hipMemcpy(cols_dev_eval, A_d->cols, A_d->nnz * sizeof(uint), hipMemcpyHostToDevice);
    hipMemcpy(vals_dev_eval, A_d->vals, A_d->nnz * sizeof(double), hipMemcpyHostToDevice);
    hipsparseSpMatDescr_t matA_d_eval;
    hipsparseStatus_t status = hipsparseCreateCsr(&matA_d_eval, A_d->nrows, A_d->ncols, A_d->nnz,
                                                  rows_dev_eval, cols_dev_eval, vals_dev_eval,
                                                  HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                                  HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (status != HIPSPARSE_STATUS_SUCCESS)
        std::cout << "hipSPARSE error code: " << int(status) << std::endl;

    // 创建向量
    double *dev_x_eval = nullptr, *dev_y_eval = nullptr, *dev_b_eval = nullptr;
    hipMalloc((void **)&dev_x_eval, A_d->ncols * sizeof(double));
    hipMalloc((void **)&dev_y_eval, A_d->nrows * sizeof(double));
    hipMalloc((void **)&dev_b_eval, A_d->nrows * sizeof(double));
    hipMemcpy(dev_x_eval, x, A_d->ncols * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(dev_b_eval, b, A_d->nrows * sizeof(double), hipMemcpyHostToDevice);
    hipsparseDnVecDescr_t vecX_d_eval, vecY_d_eval;
    hipsparseCreateDnVec(&vecX_d_eval, A_d->ncols, dev_x_eval, HIP_R_64F);
    hipsparseCreateDnVec(&vecY_d_eval, A_d->nrows, dev_y_eval, HIP_R_64F);

    // Y=AX
    hipsparseHandle_t spmv_handle_d_eval = NULL;
    hipsparseCreate(&spmv_handle_d_eval);
    hipsparseSetPointerMode(spmv_handle_d_eval, HIPSPARSE_POINTER_MODE_DEVICE);
    void *dBuffer_d_eval = NULL;
    size_t bufferSize_d_eval = 0;
    double spmv_alpha_eval, spmv_beta_eval;
    spmv_alpha_eval = 1.0;
    spmv_beta_eval = 0.0;
    double *spmv_alpha_d_eval, *spmv_beta_d_eval;
    hipMalloc(&spmv_alpha_d_eval, sizeof(double));
    hipMalloc(&spmv_beta_d_eval, sizeof(double));
    hipMemcpy(spmv_alpha_d_eval, &spmv_alpha_eval, sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(spmv_beta_d_eval, &spmv_beta_eval, sizeof(double), hipMemcpyHostToDevice);
    hipsparseSpMV_bufferSize(spmv_handle_d_eval, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                             spmv_alpha_d_eval, matA_d_eval, vecX_d_eval, spmv_beta_d_eval, vecY_d_eval, HIP_R_64F,
                             HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSize_d_eval);
    hipMalloc(&dBuffer_d_eval, bufferSize_d_eval);
    hipsparseSpMV(spmv_handle_d_eval, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                  spmv_alpha_d_eval, matA_d_eval, vecX_d_eval, spmv_beta_d_eval, vecY_d_eval, HIP_R_64F,
                  HIPSPARSE_SPMV_ALG_DEFAULT, dBuffer_d_eval);
    hipDeviceSynchronize();
    hipFree(dBuffer_d_eval);
    hipFree(spmv_alpha_d_eval);
    hipFree(spmv_beta_d_eval);

    // Y= Y - b
    double alpha_d_eval = -1.0;
    double *alpha_dev_d_eval = nullptr;
    hipMalloc(&alpha_dev_d_eval, sizeof(double));
    hipMemcpy(alpha_dev_d_eval, &alpha_d_eval, sizeof(double), hipMemcpyHostToDevice);
    hipblasHandle_t handle_d_eval;
    hipblasCreate(&handle_d_eval);
    hipblasSetPointerMode(handle_d_eval, HIPBLAS_POINTER_MODE_DEVICE);
    hipblasDaxpy(handle_d_eval, A_d->nrows, alpha_dev_d_eval, dev_b_eval, 1, dev_y_eval, 1);

    hipMemset(alpha_dev_d_eval, 0.0, sizeof(double));
    // resid = ||Y||
    hipblasDnrm2(handle_d_eval, A_d->nrows, dev_y_eval, 1, alpha_dev_d_eval);
    double resid = 0.0;
    hipMemcpy(&resid, alpha_dev_d_eval, sizeof(double), hipMemcpyDeviceToHost); // 真正的残差

    hipFree(rows_dev_eval);
    hipFree(cols_dev_eval);
    hipFree(vals_dev_eval);
    hipFree(dev_x_eval);
    hipFree(dev_y_eval);
    hipFree(dev_b_eval);
    hipsparseDestroySpMat(matA_d_eval);
    hipsparseDestroyDnVec(vecX_d_eval);
    hipsparseDestroyDnVec(vecY_d_eval);
    hipsparseDestroy(spmv_handle_d_eval);
    hipblasDestroy(handle_d_eval);
    hipFree(alpha_dev_d_eval);

    return resid/init_res;
}

int main(int argc, char *argv[])
{
    string mtx_name = argv[1];
    mtx_name.erase(mtx_name.begin(), mtx_name.begin() + mtx_name.rfind('/') + 1);
    mtx_name.erase(mtx_name.begin() + mtx_name.find('.'), mtx_name.end());

    SpM<double> A_double;                      
    loadBinToCSR(A_double, argv[1]);

    std::cout << mtx_name << ": M = " << A_double.nrows << ", N = "  << A_double.ncols << std::endl;

    std::vector<double> x_double(A_double.nrows, 0);
    std::vector<double> b_double(A_double.nrows, 1); 
    initialize(&A_double, &x_double[0], &b_double[0]);
    
    std::cout << "start running gmres" << std::endl;

    auto res = gmres(&A_double, &x_double[0], &b_double[0]); // GMRES算法核心，需重点优化

    std::get<2>(res) = validate_result(&A_double, &x_double[0], &b_double[0]); // 精度评测，计算真正残差

    std::cout << "iters = " << std::get<0>(res) << ", time = "  << std::get<1>(res) << "ms, resid = " << std::get<2>(res) << std::endl;

    std::ofstream outfile("gmres_time.txt", std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "can not open output file!" << std::endl;
        return 1;
    }
    outfile << mtx_name << " " << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << "\n";
    outfile.close();
    
    return 0;
}
