#ifndef SPARSE_MATRIX
#define SPARSE_MATRIX
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

typedef unsigned int uint;
using namespace std;

template <typename ValueType>
class SpM
{
public:
    uint nrows;
    uint ncols;
    uint nnz;
    uint *rows;
    uint *cols;
    ValueType *vals;

    SpM() : nrows(0), ncols(0), nnz(0), rows(NULL), cols(NULL), vals(NULL){
    };
    SpM(uint nrows, uint ncols, uint nnz) : nrows(nrows), ncols(ncols), nnz(nnz) {
        rows = (uint *)malloc((nrows + 1) * sizeof(uint));
        memset(rows, 0, (nrows + 1) * sizeof(uint));
        cols = (uint *)malloc((nnz) * sizeof(uint));
        memset(cols, 0, (nnz) * sizeof(uint));
        vals = (ValueType *)malloc((nnz) * sizeof(ValueType));
        memset(vals, 0, (nnz) * sizeof(ValueType));
    };
    SpM(const SpM<ValueType> &A) {
        nrows = A.nrows;
        ncols = A.ncols;
        nnz = A.nnz;
        rows = (uint *)malloc((nrows + 1) * sizeof(uint));
        cols = (uint *)malloc((nnz) * sizeof(uint));
        vals = (ValueType *)malloc((nnz) * sizeof(ValueType));
        for (unsigned int i = 0; i < nrows + 1; i++)
        {
            rows[i] = A.rows[i];
        }
        for(unsigned int p = 0; p < nnz; p++) {
            vals[p] = A.vals[p];
            cols[p] = A.cols[p];
        }
    }
    ~SpM() {
        if(rows) free(rows);
        if(cols) free(cols);
        if(vals) free(vals);
    }
};

#endif
