#ifndef __CUDA_HEADER
#define __CUDA_HEADER

// 以下宏定义开关，只选择其中一个开启
#ifndef __CUDA__
    #define __CUDA__
#endif

// #ifndef __MATRIX__
// 	#define __MATRIX__
// #endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

extern int AllocateAndSetGlobalVarArray(void** h_array, void** h_d_array, void** d_array, int arrayByteSize);
extern int AllocateAndSetGlobalVarArray_NoDeviceVal(void** h_array, void** h_d_array, int arrayByteSize);

extern int check_unexit(cudaError_t result, char const *const func, const char *const file, int const line);

#define checkCudaErrors_unexit(val) check_unexit((val), #val, __FILE__, __LINE__)

#define checkCudaStatus(val) if(val != 0) return (int)val;

//Macros
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
#define TTNPL 16384						//Total Thread Numbers per Launch 待定
#define THREAD_NUM_PER_BLOCK 256
#define X_DIMENSION 1024
#define Y_DIMENSION 1024
#define Z_DIMENSION 1024
#define R_DIMENSION 1024

#endif