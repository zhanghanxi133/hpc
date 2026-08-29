#ifndef GMRES_H
#define GMRES_H
#include <stdio.h>
#include <iostream>
#include <vector>
#include <tuple>

using RESULT=std::tuple<int, float, double>;

double validate_result(SpM<double> *A_d, double *x, double *b);
void initialize(SpM<double> *A, double *x, double *b);
void sovlerTri(int Am, int i, double *H, double *s);
void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i);
void rotation3(uint Am, double *H, double *cs, double *sn, double *s, uint i);
RESULT gmres(SpM<double> *A_d, double *_x, double *_b);
void assign2(double *a, double *s);
void fill2(uint Am, double *a, double s);
double calculateNorm(const double *vec, uint N);
void spmv(const uint *rowPtr, const uint *colInd, const double *values,
          const double *x, double *y, uint numRows);
#endif
