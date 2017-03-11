#include "opencl_utility.h"

namespace ktt
{

std::string getOpenCLEnumName(const cl_int value)
{
    switch (value)
    {
    case CL_SUCCESS:
        return std::string("CL_SUCCESS");
    case CL_DEVICE_NOT_FOUND:
        return std::string("CL_DEVICE_NOT_FOUND");
    case CL_DEVICE_NOT_AVAILABLE:
        return std::string("CL_DEVICE_NOT_AVAILABLE");
    case CL_OUT_OF_RESOURCES:
        return std::string("CL_OUT_OF_RESOURCES");
    case CL_OUT_OF_HOST_MEMORY:
        return std::string("CL_OUT_OF_HOST_MEMORY");
    case CL_INVALID_VALUE:
        return std::string("CL_INVALID_VALUE");
    case CL_INVALID_PLATFORM:
        return std::string("CL_INVALID_PLATFORM");
    case CL_INVALID_DEVICE:
        return std::string("CL_INVALID_DEVICE");
    case CL_INVALID_KERNEL_NAME:
        return std::string("CL_INVALID_KERNEL_NAME");
    case CL_INVALID_ARG_SIZE:
        return std::string("CL_INVALID_ARG_SIZE");
    case CL_INVALID_BUFFER_SIZE:
        return std::string("CL_INVALID_BUFFER_SIZE");
    default:
        return std::string("Unknown OpenCL enum");
    }
}

void checkOpenCLError(const cl_int value)
{
    if (value != CL_SUCCESS)
    {
        throw std::runtime_error("Internal OpenCL error: " + getOpenCLEnumName(value));
    }
}

void checkOpenCLError(const cl_int value, const std::string& message)
{
    if (value != CL_SUCCESS)
    {
        throw std::runtime_error("Internal OpenCL error: " + getOpenCLEnumName(value) + "\nAdditional info: " + message);
    }
}

cl_mem_flags getOpenCLMemoryType(const KernelArgumentAccessType& kernelArgumentAccessType)
{
    switch (kernelArgumentAccessType)
    {
    case KernelArgumentAccessType::READ_ONLY:
        return CL_MEM_READ_ONLY;
    case KernelArgumentAccessType::WRITE_ONLY:
        return CL_MEM_WRITE_ONLY;
    case KernelArgumentAccessType::READ_WRITE:
        return CL_MEM_READ_WRITE;
    default:
        return CL_MEM_READ_WRITE;
    }
}

} // namespace ktt
