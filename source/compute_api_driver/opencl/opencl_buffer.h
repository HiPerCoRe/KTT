#pragma once

#include <string>
#include <vector>

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenCLBuffer
{
public:
    explicit OpenCLBuffer(const cl_context context, const cl_mem_flags type, const size_t size) :
        context(context),
        type(type),
        size(size)
    {
        cl_int result;
        buffer = clCreateBuffer(context, type, size, nullptr, &result);
        checkOpenCLError(result, std::string("clCreateBuffer"));
    }

    ~OpenCLBuffer()
    {
        checkOpenCLError(clReleaseMemObject(buffer), std::string("clReleaseMemObject"));
    }

    cl_context getContext() const
    {
        return context;
    }

    cl_mem_flags getType() const
    {
        return type;
    }

    size_t getSize() const
    {
        return size;
    }

    cl_mem getBuffer() const
    {
        return buffer;
    }

private:
    cl_context context;
    cl_mem_flags type;
    size_t size;
    cl_mem buffer;
};

} // namespace ktt
