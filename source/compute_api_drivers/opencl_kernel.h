#pragma once

#include <string>

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenCLKernel
{
public:
    explicit OpenCLKernel(const cl_program program, const std::string& kernelName):
        program(program),
        kernelName(kernelName),
        argumentsCount(0)
    {
        cl_int result;
        kernel = clCreateKernel(program, &kernelName[0], &result);
        checkOpenCLError(result);
    }

    ~OpenCLKernel()
    {
        checkOpenCLError(clReleaseKernel(kernel));
    }

    void setKernelArgument(const void* buffer)
    {
        checkOpenCLError(clSetKernelArg(kernel, argumentsCount, sizeof(cl_mem), buffer));
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
