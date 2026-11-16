#ifndef TEA_CUDA_CHECK_H
#define TEA_CUDA_CHECK_H
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
inline void teaCudaCheck(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#endif
