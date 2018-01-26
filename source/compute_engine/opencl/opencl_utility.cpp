#include <stdexcept>
#include "opencl_utility.h"

namespace ktt
{

std::string getOpenclEnumName(const cl_int value)
{
    switch (value)
    {
    case CL_SUCCESS:
        return std::string("CL_SUCCESS");
    case CL_DEVICE_NOT_FOUND:
        return std::string("CL_DEVICE_NOT_FOUND");
    case CL_DEVICE_NOT_AVAILABLE:
        return std::string("CL_DEVICE_NOT_AVAILABLE");
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return std::string("CL_MEM_OBJECT_ALLOCATION_FAILURE");
    case CL_OUT_OF_RESOURCES:
        return std::string("CL_OUT_OF_RESOURCES");
    case CL_OUT_OF_HOST_MEMORY:
        return std::string("CL_OUT_OF_HOST_MEMORY");
    case CL_BUILD_PROGRAM_FAILURE:
        return std::string("CL_BUILD_PROGRAM_FAILURE");
    case CL_INVALID_VALUE:
        return std::string("CL_INVALID_VALUE");
    case CL_INVALID_PLATFORM:
        return std::string("CL_INVALID_PLATFORM");
    case CL_INVALID_DEVICE:
        return std::string("CL_INVALID_DEVICE");
    case CL_INVALID_MEM_OBJECT:
        return std::string("CL_INVALID_MEM_OBJECT");
    case CL_INVALID_BUILD_OPTIONS:
        return std::string("CL_INVALID_BUILD_OPTIONS");
    case CL_INVALID_KERNEL_NAME:
        return std::string("CL_INVALID_KERNEL_NAME");
    case CL_INVALID_ARG_INDEX:
        return std::string("CL_INVALID_ARG_INDEX");
    case CL_INVALID_ARG_VALUE:
        return std::string("CL_INVALID_ARG_VALUE");
    case CL_INVALID_ARG_SIZE:
        return std::string("CL_INVALID_ARG_SIZE");
    case CL_INVALID_WORK_GROUP_SIZE:
        return std::string("CL_INVALID_WORK_GROUP_SIZE");
    case CL_INVALID_WORK_ITEM_SIZE:
        return std::string("CL_INVALID_WORK_ITEM_SIZE");
    case CL_INVALID_BUFFER_SIZE:
        return std::string("CL_INVALID_BUFFER_SIZE");
    default:
        return std::to_string(static_cast<int>(value));
    }
}

void checkOpenclError(const cl_int value)
{
    if (value != CL_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal OpenCL error: ") + getOpenclEnumName(value));
    }
}

void checkOpenclError(const cl_int value, const std::string& message)
{
    if (value != CL_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal OpenCL error: ") + getOpenclEnumName(value) + "\nAdditional info: " + message);
    }
}

cl_mem_flags getOpenclMemoryType(const ArgumentAccessType& accessType)
{
    switch (accessType)
    {
    case ArgumentAccessType::ReadOnly:
        return CL_MEM_READ_ONLY;
    case ArgumentAccessType::WriteOnly:
        return CL_MEM_WRITE_ONLY;
    case ArgumentAccessType::ReadWrite:
        return CL_MEM_READ_WRITE;
    default:
        return CL_MEM_READ_WRITE;
    }
}

std::string getPlatformInfoString(const cl_platform_id id, const cl_platform_info info)
{
    size_t infoSize;
    checkOpenclError(clGetPlatformInfo(id, info, 0, nullptr, &infoSize));
    std::string infoString(infoSize, ' ');
    checkOpenclError(clGetPlatformInfo(id, info, infoSize, &infoString[0], nullptr));
    
    return infoString;
}

std::string getDeviceInfoString(const cl_device_id id, const cl_device_info info)
{
    size_t infoSize;
    checkOpenclError(clGetDeviceInfo(id, info, 0, nullptr, &infoSize));
    std::string infoString(infoSize, ' ');
    checkOpenclError(clGetDeviceInfo(id, info, infoSize, &infoString[0], nullptr));

    return infoString;
}

} // namespace ktt
