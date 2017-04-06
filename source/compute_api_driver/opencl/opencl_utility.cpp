#include <stdexcept>

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
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return std::string("CL_MEM_OBJECT_ALLOCATION_FAILURE");
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
    case CL_INVALID_MEM_OBJECT:
        return std::string("CL_INVALID_MEM_OBJECT");
    case CL_INVALID_BUILD_OPTIONS:
        return std::string("CL_INVALID_BUILD_OPTIONS");
    case CL_INVALID_KERNEL_NAME:
        return std::string("CL_INVALID_KERNEL_NAME");
    case CL_INVALID_ARG_SIZE:
        return std::string("CL_INVALID_ARG_SIZE");
    case CL_INVALID_WORK_GROUP_SIZE:
        return std::string("CL_INVALID_WORK_GROUP_SIZE");
    case CL_INVALID_WORK_ITEM_SIZE:
        return std::string("CL_INVALID_WORK_ITEM_SIZE");
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
        throw std::runtime_error(std::string("Internal OpenCL error: " + getOpenCLEnumName(value)));
    }
}

void checkOpenCLError(const cl_int value, const std::string& message)
{
    if (value != CL_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal OpenCL error: " + getOpenCLEnumName(value) + "\nAdditional info: " + message));
    }
}

cl_mem_flags getOpenCLMemoryType(const ArgumentMemoryType& argumentMemoryType)
{
    switch (argumentMemoryType)
    {
    case ArgumentMemoryType::ReadOnly:
        return CL_MEM_READ_ONLY;
    case ArgumentMemoryType::WriteOnly:
        return CL_MEM_WRITE_ONLY;
    case ArgumentMemoryType::ReadWrite:
        return CL_MEM_READ_WRITE;
    default:
        return CL_MEM_READ_WRITE;
    }
}

cl_ulong getKernelRunDuration(const cl_event profilingEvent)
{
    checkOpenCLError(clWaitForEvents(1, &profilingEvent));

    cl_ulong executionStart;
    cl_ulong executionEnd;
    checkOpenCLError(clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &executionStart, nullptr));
    checkOpenCLError(clGetEventProfilingInfo(profilingEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &executionEnd, nullptr));

    return executionEnd - executionStart;
}

} // namespace ktt
