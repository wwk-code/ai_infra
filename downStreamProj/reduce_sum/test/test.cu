#include<cuda_runtime.h>
#include<stdio.h>


#define DTYPE float
#define UCEIL(a,b) ((a+b-1)/b)


__global__ __launch_bounds__(256)
void sgemm_naive(int M,int K,int N,DTYPE *A, DTYPE *B, DTYPE *C, int alpha,int beta) {
    // 1D
    int tid = gridDim.x * blockIdx.x + threadIdx.x;
    int threads_per_tb = blockDim.x;
    for(int id = tid; id < M * N; id += threads_per_tb) {
        int i = id / N, j = id % N;
        if(i < M && j < N) {
            float acc = 0.0f;
            int k=0;
            #pragma unroll
            for(; k < K-3; k+=4) {
                acc += A[i + k * M] * B[k + j * K];
                acc += A[i + (k+1) * M] * B[(k+1) + j * K];
                acc += A[i + (k+2) * M] * B[(k+2) + j * K];
                acc += A[i + (k+3) * M] * B[(k+3) + j * K];
            }
            for(; k < K; k++) {
                acc += A[i + k * M] * B[k + j * K];
            }
            C[i + j * M] = alpha * acc + beta * C[i + j * M];
        }
    }
}

void launch_kernel(int M,int K,int N,DTYPE *A, DTYPE *B, DTYPE *C, int alpha,int beta) {
    int total_elements = M * N;
    int threads_per_block = 256;
    int num_of_tb = UCEIL(total_elements,threads_per_block);
    dim3 blockDim(threads_per_block);
    dim3 gridDim(num_of_tb);
    sgemm_kernel_naive<<<gridDim,blockDim>>>(M,K,N,A,B,C,alpha,beta);
    cudaDeviceSynchronize();
}


