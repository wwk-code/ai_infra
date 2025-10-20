#include "reduce_sum.cuh"

#define CUBLAS_CHECK(err) \
    if((err) != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

    
int main(int argc, char **argv) {
    
    
    return 0;
}


