#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "main.fatbin.c"
extern void __device_stub__Z17reduce_sum_kernelPKfPfi(const float *, float *, int);
extern void __device_stub__Z25final_reduce_kernel_fixedPKfPfi(const float *, float *, int);
extern void __device_stub__Z16gpu_sgemm_kerneliiifPKfS0_fPf(int, int, int, float, const float *, const float *, float, float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z17reduce_sum_kernelPKfPfi(const float *__par0, float *__par1, int __par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(const float *, float *, int))reduce_sum_kernel)));}
# 47 "csrc/include/reduce_sum.cuh"
void reduce_sum_kernel( const float *__cuda_0,float *__cuda_1,int __cuda_2)
# 47 "csrc/include/reduce_sum.cuh"
{__device_stub__Z17reduce_sum_kernelPKfPfi( __cuda_0,__cuda_1,__cuda_2);
# 74 "csrc/include/reduce_sum.cuh"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z25final_reduce_kernel_fixedPKfPfi( const float *__par0,  float *__par1,  int __par2) {  __cudaLaunchPrologue(3); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaLaunch(((char *)((void ( *)(const float *, float *, int))final_reduce_kernel_fixed))); }
# 76 "csrc/include/reduce_sum.cuh"
void final_reduce_kernel_fixed( const float *__cuda_0,float *__cuda_1,int __cuda_2)
# 76 "csrc/include/reduce_sum.cuh"
{__device_stub__Z25final_reduce_kernel_fixedPKfPfi( __cuda_0,__cuda_1,__cuda_2);
# 103 "csrc/include/reduce_sum.cuh"
}
# 1 "main.cudafe1.stub.c"
void __device_stub__Z16gpu_sgemm_kerneliiifPKfS0_fPf( int __par0,  int __par1,  int __par2,  float __par3,  const float *__par4,  const float *__par5,  float __par6,  float *__par7) {  __cudaLaunchPrologue(8); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 4UL); __cudaSetupArgSimple(__par2, 8UL); __cudaSetupArgSimple(__par3, 12UL); __cudaSetupArgSimple(__par4, 16UL); __cudaSetupArgSimple(__par5, 24UL); __cudaSetupArgSimple(__par6, 32UL); __cudaSetupArgSimple(__par7, 40UL); __cudaLaunch(((char *)((void ( *)(int, int, int, float, const float *, const float *, float, float *))gpu_sgemm_kernel))); }
# 3 "csrc/main.cu"
void gpu_sgemm_kernel( int __cuda_0,int __cuda_1,int __cuda_2,float __cuda_3,const float *__cuda_4,const float *__cuda_5,float __cuda_6,float *__cuda_7)
# 3 "csrc/main.cu"
{__device_stub__Z16gpu_sgemm_kerneliiifPKfS0_fPf( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 16 "csrc/main.cu"
}
# 1 "main.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T13) {  __nv_dummy_param_ref(__T13); __nv_save_fatbinhandle_for_managed_rt(__T13); __cudaRegisterEntry(__T13, ((void ( *)(int, int, int, float, const float *, const float *, float, float *))gpu_sgemm_kernel), _Z16gpu_sgemm_kerneliiifPKfS0_fPf, (-1)); __cudaRegisterEntry(__T13, ((void ( *)(const float *, float *, int))final_reduce_kernel_fixed), _Z25final_reduce_kernel_fixedPKfPfi, (-1)); __cudaRegisterEntry(__T13, ((void ( *)(const float *, float *, int))reduce_sum_kernel), _Z17reduce_sum_kernelPKfPfi, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
