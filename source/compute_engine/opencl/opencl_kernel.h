#pragma once

#include <string>

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenclKernel
{
public:
    explicit OpenclKernel(const cl_program program, const std::string& kernelName) :
        program(program),
        kernelName(kernelName),
        argumentsCount(0)
    {
        cl_int result;
        kernel = clCreateKernel(program, &kernelName[0], &result);
        checkOpenclError(result, "clCreateKernel");
    }

    ~OpenclKernel()
    {
        checkOpenclError(clReleaseKernel(kernel), "clReleaseKernel");
    }

    void setKernelArgumentVector(const void* buffer)
    {
        checkOpenclError(clSetKernelArg(kernel, argumentsCount, sizeof(cl_mem), buffer), "clSetKernelArg");
        argumentsCount++;
    }

    void setKernelArgumentScalar(const void* scalarValue, const size_t valueSize)
    {
        checkOpenclError(clSetKernelArg(kernel, argumentsCount, valueSize, scalarValue), "clSetKernelArg");
        argumentsCount++;
    }

    void setKernelArgumentLocal(const size_t localSizeInBytes)
    {
        checkOpenclError(clSetKernelArg(kernel, argumentsCount, localSizeInBytes, nullptr), "clSetKernelArg");
        argumentsCount++;
    }

    cl_program getProgram() const
    {
        return program;
    }

    std::string getKernelName() const
    {
        return kernelName;
    }

    cl_kernel getKernel() const
    {
        return kernel;
    }

    cl_uint getArgumentsCount() const
    {
        return argumentsCount;
    }

private:
    cl_program program;
    std::string kernelName;
    cl_kernel kernel;
    cl_uint argumentsCount;
};

} // namespace ktt
