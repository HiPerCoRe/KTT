#pragma once

#include <string>
#include <vector>

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenclBuffer
{
public:
    explicit OpenclBuffer(const cl_context context, const cl_mem_flags type, const size_t size, const size_t kernelArgumentId) :
        context(context),
        type(type),
        size(size),
        kernelArgumentId(kernelArgumentId)
    {
        cl_int result;
        buffer = clCreateBuffer(context, type, size, nullptr, &result);
        checkOpenclError(result, std::string("clCreateBuffer"));
    }

    ~OpenclBuffer()
    {
        checkOpenclError(clReleaseMemObject(buffer), std::string("clReleaseMemObject"));
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

    size_t getKernelArgumentId() const
    {
        return kernelArgumentId;
    }

private:
    cl_context context;
    cl_mem_flags type;
    size_t size;
    cl_mem buffer;
    size_t kernelArgumentId;
};

} // namespace ktt
