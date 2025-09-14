#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>

#define FLOAT float
#define INT int
#define DTYPE FLOAT

#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))

#define A(i,j) A[i + j * lda]
#define B(i,j) B[i + j * ldb]
#define C(i,j) C[i + j * ldc]


#define CUDA_CALLER(call) do { \
    cudaError_t cuda_ret = (call); \
    if(cuda_ret != cudaSuccess) { \
        printf("CUDA Error at line %d in file %s \n",__LINE__,__FILE__); \
        printf("  Error message: %s\n", cudaGetErrorString(cuda_ret)); \
        printf(" In the function call %s \n",#call); \
        exit(1); \
    } \
} while(0)



// 矩阵零初始化函数
void matrix_init(DTYPE *A,int m,int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i + j * m] = 0;
        }
    }
}

// 矩阵随机初始化函数
void randomize_matrix(DTYPE *mat, int m,int n) {
    srand(time(NULL));
    for(int i=0;i<m;i++) {
        for(int j=0;j<n;j++) {
            DTYPE tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
            tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
            mat[i + j * m] = tmp;
        }
    }
}

// 自定义的矩阵拷贝函数
void copy_matrix(DTYPE *src, DTYPE *dest, int m,int n){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            *(dest + i + j * m) = *(src + i + j * m);
        }
    }
}


// 比较两个矩阵是否相等
bool verify_cmp_matrix(DTYPE *mat1, DTYPE *mat2, int m,int n, double limit=1e-2) {
    double diff = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            diff = fabs((double)mat1[i + j * m] - (double)mat2[i + j * m]);    
            if(diff > limit) {
                printf("error. %5.2f,%5.2f,[i,j]: %d,%d\n", mat1[i + j * m],mat2[i + j * m],i,j);
                return false;
            }
        }
    }
    return true;
}


