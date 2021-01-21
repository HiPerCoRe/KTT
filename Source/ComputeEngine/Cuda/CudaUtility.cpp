#ifdef KTT_API_CUDA

#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

std::string GetEnumName(const CUresult value)
{
    const char* name;
    cuGetErrorName(value, &name);
    return name;
}

std::string GetEnumName(const nvrtcResult value)
{
    const char* name = nvrtcGetErrorString(value);
    return name;
}

void CheckError(const CUresult value, const std::string& function, const std::string& info)
{
    if (value != CUDA_SUCCESS)
    {
        throw KttException("CUDA engine encountered error " + GetEnumName(value) + " in function " + function
            + ", additional info: " + info);
    }
}

void CheckError(const nvrtcResult value, const std::string& function, const std::string& info)
{
    if (value != NVRTC_SUCCESS)
    {
        throw KttException("CUDA NVRTC encountered error " + GetEnumName(value) + " in function " + function
            + ", additional info: " + info);
    }
}

#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)

std::string GetEnumName(const CUptiResult value)
{
    const char* name;
    cuptiGetResultString(value, &name);
    return name;
}

void CheckError(const CUptiResult value, const std::string& function, const std::string& info)
{
    if (value != CUPTI_SUCCESS)
    {
        throw KttException("CUDA CUPTI encountered error " + GetEnumName(value) + " in function " + function
            + ", additional info: " + info);
    }
}

#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI

} // namespace ktt

#endif // KTT_API_CUDA
