#include "common.cuh"

// 简单的CPU矩阵乘法实现
void cpu_sgemm(int m, int k, int n, FLOAT alpha, const DTYPE* A, const DTYPE* B, FLOAT beta, DTYPE* C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            FLOAT sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i + p * m] * B[p + j * k];
            }
            C[i + j * m] = alpha * sum + beta * C[i + j * m];
        }
    }
}

// GPU矩阵乘法kernel（简单实现）
__global__ void gpu_sgemm_kernel(int m, int k, int n, FLOAT alpha, const DTYPE* A, const DTYPE* B, FLOAT beta, DTYPE* C);
// __global__ void gpu_sgemm_kernel(int m, int k, int n, FLOAT alpha, const DTYPE* A, const DTYPE* B, FLOAT beta, DTYPE* C) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total = m * n;
    
//     if (idx < total) {
//         int i = idx % m;
//         int j = idx / m;
//         FLOAT sum = 0.0f;
//         for (int p = 0; p < k; p++) {
//             sum += A[i + p * m] * B[p + j * k];
//         }
//         C[idx] = alpha * sum + beta * C[idx];
//     }
// }

// 测试kernel调度函数
void test_kernel(int kernel_id, int m, int k, int n, FLOAT alpha, const DTYPE* A, const DTYPE* B, FLOAT beta, DTYPE* C, cublasHandle_t handle) {
    if (kernel_id == 0) {
        // CPU实现
        cpu_sgemm(m, k, n, alpha, A, B, beta, C);
    } else {
        // GPU实现
        int total = m * n;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        gpu_sgemm_kernel<<<blocks, threads>>>(m, k, n, alpha, A, B, beta, C);
    }
}

__global__ void reduce_sum_kernel(const DTYPE *g_idata, DTYPE *g_odata, int n) {
    const int tid_in_block = threadIdx.x;
    const int block_size = blockDim.x;
    const int block_start_idx = blockIdx.x * block_size;

    __shared__ DTYPE sdata[256];

    int idx = block_start_idx + tid_in_block;

    sdata[tid_in_block] = 0.0;
    while (idx < n) {
        sdata[tid_in_block] += g_idata[idx];
        idx += gridDim.x * block_size;
    }
    
    __syncthreads();

    for (int s = block_size / 2; s > 0; s /= 2) {
        if (tid_in_block < s) {
            sdata[tid_in_block] += sdata[tid_in_block + s];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void final_reduce_kernel_fixed(const DTYPE *g_idata_partial, DTYPE *g_odata_final, int n_elements_partial) {
    const int tid_in_block = threadIdx.x;
    const int block_size = blockDim.x;
    const int block_start_idx = blockIdx.x * block_size;
    __shared__ DTYPE sdata[256];

    int idx = block_start_idx + tid_in_block;
    sdata[tid_in_block] = 0.0;
    if (idx < n_elements_partial) {
        sdata[tid_in_block] = g_idata_partial[idx];
    }
    
    __syncthreads();

    // 线程块内部的归约逻辑不变
    for (int s = block_size / 2; s > 0; s /= 2) {
        if (tid_in_block < s) {
            sdata[tid_in_block] += sdata[tid_in_block + s];
        }
        __syncthreads();
    }

    // 使用 atomicAdd() 来进行线程块间的最终累加
    if (tid_in_block == 0) {
        // 将本线程块的局部和原子性地加到最终结果地址
        atomicAdd(g_odata_final, sdata[0]);
    }
}


void host_reduce_sum(const DTYPE *h_idata, DTYPE *h_odata, int n_elements) {
    // 线程块大小
    const int threads_per_block = 256;
    
    //----------------------------------------------------
    // 第一阶段：启动 reduce_sum_kernel
    //----------------------------------------------------
    
    // 计算第一阶段的网格大小
    int num_blocks_1 = CEIL_DIV(n_elements, threads_per_block);
    
    // 分配设备内存
    DTYPE *d_idata = nullptr;
    DTYPE *d_odata_partial = nullptr;
    DTYPE *d_odata_final = nullptr;

    cudaMalloc(&d_idata, n_elements * sizeof(DTYPE));
    cudaMalloc(&d_odata_partial, num_blocks_1 * sizeof(DTYPE));
    
    // 从主机端拷贝数据到设备端
    cudaMemcpy(d_idata, h_idata, n_elements * sizeof(DTYPE), cudaMemcpyHostToDevice);

    // 启动第一个 Kernel
    reduce_sum_kernel<<<num_blocks_1, threads_per_block>>>(d_idata, d_odata_partial, n_elements);
    cudaDeviceSynchronize();
    
    //----------------------------------------------------
    // 第二阶段：启动 final_reduce_kernel
    //----------------------------------------------------
    
    // 计算第二阶段的网格大小
    int n_elements_partial = num_blocks_1;
    int num_blocks_2 = CEIL_DIV(n_elements_partial, threads_per_block);
    
    // 分配最终结果的设备内存（只需要一个元素）
    cudaMalloc(&d_odata_final, sizeof(DTYPE));
    cudaMemset(d_odata_final, 0, sizeof(DTYPE));
    
    // 启动第二个 Kernel
    final_reduce_kernel_fixed<<<num_blocks_2, threads_per_block>>>(d_odata_partial, d_odata_final, n_elements_partial);
    cudaDeviceSynchronize();
    
    // 将最终结果从设备端拷贝回主机端
    cudaMemcpy(h_odata, d_odata_final, sizeof(DTYPE), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_idata);
    cudaFree(d_odata_partial);
    cudaFree(d_odata_final);
}