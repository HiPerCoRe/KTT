#ifdef PLATFORM_CUDA

#include <stdexcept>

#include "cuda_utility.h"

namespace ktt
{

void checkCudaError(const CUresult value)
{
    if (value != CUDA_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA error: ") + std::to_string(value));
    }
}

void checkCudaError(const CUresult value, const std::string& message)
{
    if (value != CUDA_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA error: ") + std::to_string(value) + "\nAdditional info: " + message);
    }
}

} // namespace ktt

#endif // PLATFORM_CUDA
