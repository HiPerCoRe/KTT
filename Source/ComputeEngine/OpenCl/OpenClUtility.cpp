#ifdef KTT_API_OPENCL

#include <Api/KttException.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>

namespace ktt
{

std::string GetEnumName(const cl_int value)
{
    switch (value)
    {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    default:
        return std::to_string(static_cast<int>(value));
    }
}

void CheckError(const cl_int value, const std::string& function, const std::string& info)
{
    if (value != CL_SUCCESS)
    {
        throw KttException("OpenCL engine encountered error " + GetEnumName(value) + " in function " + function
            + ", additional info: " + info);
    }
}

#if defined(KTT_PROFILING_GPA) || defined(KTT_PROFILING_GPA_LEGACY)
void CheckError(const GPA_Status value, GPAFunctionTable& functions, const std::string& function, const std::string& info)
{
    if (value != GPA_STATUS_OK)
    {
        throw KttException(std::string("OpenCL GPA profiling engine encountered error ") + functions.GPA_GetStatusAsStr(value)
            + " in function " + function + ", additional info: " + info);
    }
}
#endif // KTT_PROFILING_GPA || KTT_PROFILING_GPA_LEGACY

} // namespace ktt

#endif // KTT_API_OPENCL
