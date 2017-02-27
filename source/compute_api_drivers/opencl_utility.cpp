#include "opencl_utility.h"

namespace ktt
{

std::string getOpenCLEnumName(const cl_int value)
{
    switch (value)
    {
    case CL_SUCCESS:
        return std::string("CL_SUCCESS");
    case CL_INVALID_PLATFORM:
        return std::string("CL_INVALID_PLATFORM");
    case CL_INVALID_DEVICE:
        return std::string("CL_INVALID_DEVICE");
    case CL_INVALID_VALUE:
        return std::string("CL_INVALID_VALUE");
    case CL_OUT_OF_RESOURCES:
        return std::string("CL_OUT_OF_RESOURCES");
    case CL_OUT_OF_HOST_MEMORY:
        return std::string("CL_OUT_OF_HOST_MEMORY");
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

} // namespace ktt
