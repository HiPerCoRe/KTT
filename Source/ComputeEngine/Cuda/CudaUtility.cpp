#ifdef KTT_API_CUDA

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CudaUtility.h>

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
    if (value == CUDA_SUCCESS)
    {
        return;
    }

    std::string message = "CUDA engine encountered error " + GetEnumName(value) + " in function " + function;

    if (!info.empty())
    {
        message += ", additional info: " + info;
    }

    throw KttException(message);
}

void CheckError(const nvrtcResult value, const std::string& function, const std::string& info)
{
    if (value == NVRTC_SUCCESS)
    {
        return;
    }

    std::string message = "CUDA NVRTC encountered error " + GetEnumName(value) + " in function " + function;

    if (!info.empty())
    {
        message += ", additional info: " + info;
    }

    throw KttException(message);
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
    if (value == CUPTI_SUCCESS)
    {
        return;
    }

    std::string message = "CUDA CUPTI encountered error " + GetEnumName(value) + " in function " + function;

    if (!info.empty())
    {
        message += ", additional info: " + info;
    }

    throw KttException(message);
}

#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI

#if defined(KTT_PROFILING_CUPTI)

std::string GetEnumName(const NVPA_Status value)
{
    switch (value)
    {
    case NVPA_STATUS_SUCCESS:
        return "NVPA_STATUS_SUCCESS";
    case NVPA_STATUS_ERROR:
        return "NVPA_STATUS_ERROR";
    case NVPA_STATUS_INTERNAL_ERROR:
        return "NVPA_STATUS_INTERNAL_ERROR";
    case NVPA_STATUS_NOT_INITIALIZED:
        return "NVPA_STATUS_NOT_INITIALIZED";
    case NVPA_STATUS_NOT_LOADED:
        return "NVPA_STATUS_NOT_LOADED";
    case NVPA_STATUS_FUNCTION_NOT_FOUND:
        return "NVPA_STATUS_FUNCTION_NOT_FOUND";
    case NVPA_STATUS_NOT_SUPPORTED:
        return "NVPA_STATUS_NOT_SUPPORTED";
    case NVPA_STATUS_NOT_IMPLEMENTED:
        return "NVPA_STATUS_NOT_IMPLEMENTED";
    case NVPA_STATUS_INVALID_ARGUMENT:
        return "NVPA_STATUS_INVALID_ARGUMENT";
    case NVPA_STATUS_INVALID_METRIC_ID:
        return "NVPA_STATUS_INVALID_METRIC_ID";
    case NVPA_STATUS_DRIVER_NOT_LOADED:
        return "NVPA_STATUS_DRIVER_NOT_LOADED";
    case NVPA_STATUS_OUT_OF_MEMORY:
        return "NVPA_STATUS_OUT_OF_MEMORY";
    case NVPA_STATUS_INVALID_THREAD_STATE:
        return "NVPA_STATUS_INVALID_THREAD_STATE";
    case NVPA_STATUS_FAILED_CONTEXT_ALLOC:
        return "NVPA_STATUS_FAILED_CONTEXT_ALLOC";
    case NVPA_STATUS_UNSUPPORTED_GPU:
        return "NVPA_STATUS_UNSUPPORTED_GPU";
    case NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION:
        return "NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION";
    default:
        return std::to_string(static_cast<int>(value));
    }
}

void CheckError(const NVPA_Status value, const std::string& function, const std::string& info)
{
    if (value == NVPA_STATUS_SUCCESS)
    {
        return;
    }

    std::string message = "CUDA NVPA encountered error " + GetEnumName(value) + " in function " + function;

    if (!info.empty())
    {
        message += ", additional info: " + info;
    }

    throw KttException(message);
}

#endif // KTT_PROFILING_CUPTI

} // namespace ktt

#endif // KTT_API_CUDA
