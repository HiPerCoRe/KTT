#pragma once

#include <string>
#include <CL/cl.h>
#include <api/kernel_compilation_data.h>
#include <compute_engine/opencl/opencl_utility.h>

namespace ktt
{

class OpenCLKernel
{
public:
    explicit OpenCLKernel(const cl_device_id device, const cl_program program, const std::string& kernelName) :
        device(device),
        program(program),
        kernelName(kernelName),
        argumentsCount(0)
    {
        cl_int result;
        kernel = clCreateKernel(program, &kernelName[0], &result);
        checkOpenCLError(result, "clCreateKernel");
    }

    ~OpenCLKernel()
    {
        checkOpenCLError(clReleaseKernel(kernel), "clReleaseKernel");
    }

    void setKernelArgumentVector(const void* buffer)
    {
        checkOpenCLError(clSetKernelArg(kernel, argumentsCount, sizeof(cl_mem), buffer), "clSetKernelArg");
        argumentsCount++;
    }

    void setKernelArgumentVectorSVM(const void* buffer)
    {
        #ifdef CL_VERSION_2_0
        checkOpenCLError(clSetKernelArgSVMPointer(kernel, argumentsCount, buffer), "clSetKernelArgSVMPointer");
        argumentsCount++;
        #else
        throw std::runtime_error("Unified memory buffers are not supported on this platform");
        #endif
    }

    void setKernelArgumentScalar(const void* scalarValue, const size_t valueSize)
    {
        checkOpenCLError(clSetKernelArg(kernel, argumentsCount, valueSize, scalarValue), "clSetKernelArg");
        argumentsCount++;
    }

    void setKernelArgumentLocal(const size_t localSizeInBytes)
    {
        checkOpenCLError(clSetKernelArg(kernel, argumentsCount, localSizeInBytes, nullptr), "clSetKernelArg");
        argumentsCount++;
    }

    void resetKernelArguments()
    {
        argumentsCount = 0;
    }

    cl_device_id getDevice() const
    {
        return device;
    }

    cl_program getProgram() const
    {
        return program;
    }

    const std::string& getKernelName() const
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

    KernelCompilationData getCompilationData() const
    {
        KernelCompilationData result;

        collectAttribute(result.maxWorkGroupSize, CL_KERNEL_WORK_GROUP_SIZE);
        collectAttribute(result.localMemorySize, CL_KERNEL_LOCAL_MEM_SIZE);
        collectAttribute(result.privateMemorySize, CL_KERNEL_PRIVATE_MEM_SIZE);
        // It is currently not possible to retrieve remaining compilation data attributes in OpenCL

        return result;
    }

private:
    cl_device_id device;
    cl_program program;
    std::string kernelName;
    cl_kernel kernel;
    cl_uint argumentsCount;

    void collectAttribute(uint64_t& output, const cl_kernel_work_group_info attribute) const
    {
        checkOpenCLError(clGetKernelWorkGroupInfo(kernel, device, attribute, sizeof(uint64_t), &output, nullptr), "clGetKernelWorkGroupInfo");
    }
};

} // namespace ktt
