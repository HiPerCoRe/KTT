#pragma once

#include <string>
#include <vector>

#include "CL/cl.h"
#include "opencl_utility.h"

namespace ktt
{

class OpenCLProgram
{
public:
    explicit OpenCLProgram(const std::string source, const cl_context context, const std::vector<cl_device_id>& devices):
        source(source),
        context(context),
        devices(devices)
    {
        cl_int result;
        size_t sourceLength = source.size();
        auto sourcePointer = &source[0];
        program = clCreateProgramWithSource(context, 1, &sourcePointer, &sourceLength, &result);
        checkOpenCLError(result);
    }

    ~OpenCLProgram()
    {
        checkOpenCLError(clReleaseProgram(program));
    }

    cl_context getContext() const
    {
        return context;
    }

    std::string getSource() const
    {
        return source;
    }

    std::vector<cl_device_id> getDevices() const
    {
        return devices;
    }

    cl_program getProgram() const
    {
        return program;
    }

private:
    std::string source;
    cl_context context;
    std::vector<cl_device_id> devices;
    cl_program program;
};

} // namespace ktt
