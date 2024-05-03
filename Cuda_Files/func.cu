#include <iostream>
#include "func.h"
#include "common.h"
#include "CUDAHeader.cuh"

#ifdef __CUDA__
	int check_unexit(cudaError_t result, char const *const func, const char *const file, int const line) {
			if (result) {
				fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
				//DEVICE_RESET
			}
			// Make sure we call CUDA Device Reset before exiting
			return (int)result;
		}

	int AllocateAndSetGlobalVarArray(void** h_array, void** h_d_array, void** d_array, int arrayByteSize)
	{
		int result = 0;
		result = checkCudaErrors_unexit(cudaMalloc(h_d_array, arrayByteSize));
		if (result != 0)
			return result;

		result = checkCudaErrors_unexit(cudaMemcpy(*h_d_array, *h_array, arrayByteSize, cudaMemcpyHostToDevice));
		result = checkCudaErrors_unexit(cudaMemcpyToSymbol(*d_array, h_d_array, sizeof((*h_d_array))));
		return result;
	}

	int AllocateAndSetGlobalVarArray_NoDeviceVal(void** h_array, void** h_d_array, int arrayByteSize)
	{
		int result = 0;
		result = checkCudaErrors_unexit(cudaMalloc(h_d_array, arrayByteSize));
		if (result != 0)
			return result;

		result = checkCudaErrors_unexit(cudaMemcpy(*h_d_array, *h_array, arrayByteSize, cudaMemcpyHostToDevice));
		return result;
	}
#endif