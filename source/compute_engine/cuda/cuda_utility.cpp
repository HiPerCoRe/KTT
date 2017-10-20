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

std::string getNvrtcEnumName(const nvrtcResult value)
{
    switch (value)
    {
    case NVRTC_SUCCESS:
        return std::string("NVRTC_SUCCESS");
    case NVRTC_ERROR_OUT_OF_MEMORY:
        return std::string("NVRTC_ERROR_OUT_OF_MEMORY");
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
        return std::string("NVRTC_ERROR_PROGRAM_CREATION_FAILURE");
    case NVRTC_ERROR_INVALID_INPUT:
        return std::string("NVRTC_ERROR_INVALID_INPUT");
    case NVRTC_ERROR_INVALID_PROGRAM:
        return std::string("NVRTC_ERROR_INVALID_PROGRAM");
    case NVRTC_ERROR_INVALID_OPTION:
        return std::string("NVRTC_ERROR_INVALID_OPTION");
    case NVRTC_ERROR_COMPILATION:
        return std::string("NVRTC_ERROR_COMPILATION");
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
        return std::string("NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID");
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

void checkCudaError(const nvrtcResult value, const std::string& message)
{
    if (value != NVRTC_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA NVRTC error: ") + getNvrtcEnumName(value) + "\nAdditional info: " + message);
    }
}

float getKernelRunDuration(const CUevent start, const CUevent end)
{
    float result;
    checkCudaError(cuEventElapsedTime(&result, start, end), "cuEventElapsedTime");

    return result * 1'000'000.0f; // return duration in nanoseconds
}

} // namespace ktt

#endif // PLATFORM_CUDA
