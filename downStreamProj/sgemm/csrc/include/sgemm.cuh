#include "common.cuh"


// CPU矩阵乘函数(列优先版)
void matrix_multiply(int M, int K, int N, DTYPE *A, DTYPE *B, DTYPE *C, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i + k * M] * B[k + j * K];
            }
            C[i + j * M] = alpha * sum + beta * C[i + j * M];
        }
    }
}


// naive kernel (column major)
__global__ __launch_bounds__(1024)
void sgemm_kernel_naive_1(int M,int N,int K,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;   // 纵向
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;   // 横向
    if(x < M && y < N) {
        float temp = 0.0;
        for(int k = 0; k < K; ++k) {
            temp += A[x + M * k] * B[k + y * K];
        }
        C[x + M * y] = alpha * temp + beta * C[x + M * y];
    }
}


void test_sgemm_kernel_naive_1(int M,int K,int N,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    cudaDeviceSynchronize();
    dim3 blockDim(32,32,1);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(K,32),1);
    sgemm_kernel_naive_1<<<gridDim,blockDim>>>(M,N,K,A,B,C,alpha,beta);
    cudaDeviceSynchronize();
}


// naive kernel (column major)
__global__ __launch_bounds__(1024)
void sgemm_kernel_naive_2(int M,int K,int N,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    // 使用1D线程索引，更灵活地处理任意矩阵大小
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    
    for (int idx = tid; idx < M * N; idx += total_threads) {
        int i = idx / N;  // 行索引 - 修复：应该是 idx / N
        int j = idx % N;  // 列索引 - 修复：应该是 idx % N
        
        if (i < M && j < N) {
            DTYPE sum = 0.0;
            // 循环展开优化，每次处理4个元素
            int k = 0;
            #pragma unroll
            for (; k < K - 3; k += 4) {
                sum += A[i + k * M] * B[k + j * K];
                sum += A[i + (k+1) * M] * B[(k+1) + j * K];
                sum += A[i + (k+2) * M] * B[(k+2) + j * K];
                sum += A[i + (k+3) * M] * B[(k+3) + j * K];
            }
            // 处理剩余的元素
            for (; k < K; ++k) {
                sum += A[i + k * M] * B[k + j * K];
            }
            C[i + j * M] = alpha * sum + beta * C[i + j * M];
        }
    }
}

void test_sgemm_kernel_naive_2(int M,int K,int N,DTYPE *A,DTYPE *B,DTYPE *C,float alpha,float beta) {
    // 动态计算线程块大小，确保有足够的线程处理所有元素
    int total_elements = M * N;
    int threads_per_block = 1024;
    int num_blocks = CEIL_DIV(total_elements, threads_per_block);
    
    // 限制最大线程块数量，避免资源浪费
    num_blocks = min(num_blocks, 65535);
    
    dim3 blockDim(threads_per_block);
    dim3 gridDim(num_blocks);
    
    sgemm_kernel_naive_2<<<gridDim,blockDim>>>(M,K,N,A,B,C,alpha,beta);
    cudaDeviceSynchronize();
}


#define BM 64    
#define BN 64    
#define BK 8     
#define TM 8     
#define TN 8  
__global__ void sgemm_optimized(int M,int K,int N, float* A, float *B, float *C, float alpha, float beta) {
    
    __shared__ float As[BM][BK];  
    __shared__ float Bs[BK][BN];  
    
    float c[TM][TN] = {{0.0f}};
    
    // 块索引计算
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    
    // 线程索引计算（每个线程处理8x8子矩阵）
    int thread_row = threadIdx.y * TM;  
    int thread_col = threadIdx.x * TN;  
    
    // 动态指针偏移
    A += block_row * K;  
    B += block_col;      
    C += block_row * N + block_col;

    // 主循环处理K维度（分阶段加载数据）外层 Split-K
    for(int t = 0; t < K; t += BK) {
        #pragma unroll
        for(int i=0; i<BM; i+=blockDim.y) { // 一个线程负责 BM / TM 个 C元素复制
            if(block_row+i < M && t+threadIdx.x < K) 
                As[i+threadIdx.y][threadIdx.x] = A[(i+threadIdx.y)*K + t + threadIdx.x];
        }
        #pragma unroll
        for(int j=0; j<BN; j+=blockDim.x) {
            if(block_col+j < N && t+threadIdx.y < K)
                Bs[threadIdx.y][j+threadIdx.x] = B[(t+threadIdx.y)*N + j+threadIdx.x];
        }
        __syncthreads();  // 同步TB

        // 核心计算部分（循环展开优化）
        #pragma unroll
        for(int k=0; k<BK; ++k) {  // 内层 Split-BK

            float a_reg[TM], b_reg[TN];
            
            // 寄存器预加载（减少Shared Memory访问次数）
            #pragma unroll
            for(int i=0; i<TM; ++i) 
                a_reg[i] = As[thread_row+i][k];
                
            #pragma unroll
            for(int j=0; j<TN; ++j) 
                b_reg[j] = Bs[k][thread_col+j];
            
            // 外积计算（TMxTN子矩阵乘加）
            #pragma unroll
            for(int i=0; i<TM; ++i) {
                #pragma unroll
                for(int j=0; j<TN; ++j) {
                    c[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // 结果写回全局内存（边界处理）
    #pragma unroll
    for(int i=0; i<TM; ++i) {
        #pragma unroll
        for(int j=0; j<TN; ++j) {
            int row = thread_row + i;
            int col = thread_col + j;
            if(row < M && col < N) {
                C[row*N + col] = alpha * c[i][j] + beta * C[row*N + col];
            }
        }
    }
}


void launch_sgemm_shared(int M,int K,int N, float* A, float* B, float* C, float alpha, float beta) {
    dim3 block(BN/TN , BM/TM);  // (64/8)x(64/8)=8x8=64线程/block
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    sgemm_optimized<<<grid, block>>>(M, N, K, A, B, C, alpha, beta);
}


void test_kernel(int kernel_num,int M,int K,int N,DTYPE alpha,DTYPE *A,DTYPE *B,DTYPE beta,DTYPE *C, cublasHandle_t handle){
    switch (kernel_num){    
        
        case 0: matrix_multiply(M,K,N,A,B,C,alpha,beta); break;
        case 1: test_sgemm_kernel_naive_1(M,K,N,A,B,C,alpha,beta); break;
        case 2: test_sgemm_kernel_naive_2(M,K,N,A,B,C,alpha,beta); break;
        case 3: launch_sgemm_shared(M,K,N,A,B,C,alpha,beta); break;

        default: break;
    }
}

