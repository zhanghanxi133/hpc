#include <iostream>
#include <vector>
#include <fstream>
#include "sparseMatrix.hpp"
#include "gmres.hpp"

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

int main(int argc, char *argv[])
{
    string mtx_name = argv[1];
    mtx_name.erase(mtx_name.begin(), mtx_name.begin() + mtx_name.rfind('/') + 1);
    mtx_name.erase(mtx_name.begin() + mtx_name.find('.'), mtx_name.end());

    SpM<double> A_double;                      //禁止修改
    loadBinToCSR(A_double, argv[1]); //禁止修改

    std::cout << mtx_name << ": M = " << A_double.nrows << ", N = "  << A_double.ncols << std::endl;

    std::vector<double> x_double(A_double.nrows, 0); //禁止修改
    std::vector<double> b_double(A_double.nrows, 1); //禁止修改
    initialize(&A_double, &x_double[0], &b_double[0]);

    std::cout << "start running gmres" << std::endl;
    auto res = gmres(&A_double, &x_double[0], &b_double[0]); // GMRES算法核心，需重点优化

    std::cout << "iters = " << std::get<0>(res) << ", time = "  << std::get<1>(res) << "ms, resid = " << std::get<2>(res) << std::endl;

    // ==============禁止修改下列结果输出==============
    std::ofstream outfile("gmres_time.txt", std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "can not open output file!" << std::endl;
        return 1;
    }
    outfile << mtx_name << " " << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << "\n";
    outfile.close();
    // ==============禁止修改上列结果输出==============
    
    return 0;
}
