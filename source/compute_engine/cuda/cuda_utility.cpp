#ifdef PLATFORM_CUDA

#include <stdexcept>
#include "cuda_utility.h"

namespace ktt
{

std::string getCudaEnumName(const CUresult value)
{
    const char* name;
    cuGetErrorName(value, &name);
    return name;
}

std::string getNvrtcEnumName(const nvrtcResult value)
{
    std::string name = nvrtcGetErrorString(value);
    return name;
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

float getEventCommandDuration(const CUevent start, const CUevent end)
{
    float result;
    checkCudaError(cuEventElapsedTime(&result, start, end), "cuEventElapsedTime");

    return result * 1'000'000.0f; // return duration in nanoseconds
}

} // namespace ktt

#endif // PLATFORM_CUDA
