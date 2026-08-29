#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <hip/hip_runtime.h>
#include <float.h>
#include <fstream>

#include <assert.h>
#include <stdio.h>

#define HIP_CHECK(status) \
    do { \
        hipError_t err = status; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error: %s at line %d in file %s\n", \
                    hipGetErrorString(err), __LINE__, __FILE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CEIL_DIV(a, b) ((a + b - 1) / b)

extern "C" void solve(const float* input, float* output, int N);

#endif 