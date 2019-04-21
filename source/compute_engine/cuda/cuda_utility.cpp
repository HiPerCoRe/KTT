#ifdef KTT_PLATFORM_CUDA

#include <stdexcept>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

std::string getCUDAEnumName(const CUresult value)
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

void checkCUDAError(const CUresult value)
{
    if (value != CUDA_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA error: ") + getCUDAEnumName(value));
    }
}

void checkCUDAError(const CUresult value, const std::string& message)
{
    if (value != CUDA_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA error: ") + getCUDAEnumName(value) + "\nAdditional info: " + message);
    }
}

void checkCUDAError(const nvrtcResult value, const std::string& message)
{
    if (value != NVRTC_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA NVRTC error: ") + getNvrtcEnumName(value) + "\nAdditional info: " + message);
    }
}

float getEventCommandDuration(const CUevent start, const CUevent end)
{
    float result;
    checkCUDAError(cuEventElapsedTime(&result, start, end), "cuEventElapsedTime");

    return result * 1'000'000.0f; // return duration in nanoseconds
}

#ifdef KTT_PROFILING
std::string getCUPTIEnumName(const CUptiResult value)
{
    const char* name;
    cuptiGetResultString(value, &name);
    return name;
}

void checkCUDAError(const CUptiResult value, const std::string& message)
{
    if (value != CUPTI_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUDA CUPTI error: ") + getCUPTIEnumName(value) + "\nAdditional info: " + message);
    }
}
#endif // KTT_PROFILING

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
