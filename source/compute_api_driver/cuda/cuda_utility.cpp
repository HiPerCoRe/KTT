#ifdef PLATFORM_CUDA

#include <stdexcept>

#include "cuda_utility.h"

namespace ktt
{

std::string getCudaEnumName(const CUresult value)
{
    switch (value)
    {
    case CUDA_SUCCESS:
        return std::string("CUDA_SUCCESS");
    case CUDA_ERROR_INVALID_VALUE:
        return std::string("CUDA_ERROR_INVALID_VALUE");
    case CUDA_ERROR_OUT_OF_MEMORY:
        return std::string("CUDA_ERROR_OUT_OF_MEMORY");
    case CUDA_ERROR_NO_DEVICE:
        return std::string("CUDA_ERROR_NO_DEVICE");
    case CUDA_ERROR_INVALID_SOURCE:
        return std::string("CUDA_ERROR_INVALID_SOURCE");
    case CUDA_ERROR_FILE_NOT_FOUND:
        return std::string("CUDA_ERROR_FILE_NOT_FOUND");
    case CUDA_ERROR_INVALID_HANDLE:
        return std::string("CUDA_ERROR_INVALID_HANDLE");
    case CUDA_ERROR_ILLEGAL_ADDRESS:
        return std::string("CUDA_ERROR_ILLEGAL_ADDRESS");
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
        return std::string("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES");
    case CUDA_ERROR_LAUNCH_FAILED:
        return std::string("CUDA_ERROR_LAUNCH_FAILED");
    default:
        return std::to_string(static_cast<int>(value));
    }
}

void checkCudaError(const CUresult value)
{
    if (value != CUDA_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA error: ") + getCudaEnumName(value));
    }
}

void checkCudaError(const CUresult value, const std::string& message)
{
    if (value != CUDA_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA error: ") + getCudaEnumName(value) + "\nAdditional info: " + message);
    }
}

float getKernelRunDuration(const CUevent start, const CUevent end)
{
    // to do
    return 0.0f;
}

} // namespace ktt

#endif // PLATFORM_CUDA
