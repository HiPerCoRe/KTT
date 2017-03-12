#include "opencl_core.h"

#include "CL/cl.h"

namespace ktt
{

OpenCLCore::OpenCLCore(const size_t platformIndex, const std::vector<size_t>& deviceIndices):
    compilerOptions(std::string(""))
{
    auto platforms = getOpenCLPlatforms();
    if (platformIndex >= platforms.size())
    {
        throw std::runtime_error("Invalid platform index: " + platformIndex);
    }

    auto devices = getOpenCLDevices(platforms.at(platformIndex));
    std::vector<cl_device_id> deviceIds;
    for (const auto deviceIndex : deviceIndices)
    {
        if (deviceIndex >= devices.size())
        {
            throw std::runtime_error("Invalid device index: " + deviceIndex);
        }
        deviceIds.push_back(devices.at(deviceIndex).getId());
    }

    context = std::make_unique<OpenCLContext>(platforms.at(platformIndex).getId(), deviceIds);
    for (const auto deviceIndex : context->getDevices())
    {
        commandQueues.push_back(std::make_unique<OpenCLCommandQueue>(context->getContext(), deviceIndex));
    }
}

std::vector<OpenCLPlatform> OpenCLCore::getOpenCLPlatforms()
{
    cl_uint platformCount;
    checkOpenCLError(clGetPlatformIDs(0, nullptr, &platformCount));

    std::vector<cl_platform_id> platformIds(platformCount);
    checkOpenCLError(clGetPlatformIDs(platformCount, platformIds.data(), nullptr));

    std::vector<OpenCLPlatform> platforms;
    for (const auto platformId : platformIds)
    {
        std::string name = getPlatformInfo(platformId, CL_PLATFORM_NAME);
        platforms.push_back(OpenCLPlatform(platformId, name));
    }

    return platforms;
}

std::vector<OpenCLDevice> OpenCLCore::getOpenCLDevices(const OpenCLPlatform& platform)
{
    cl_uint deviceCount;
    checkOpenCLError(clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount));

    std::vector<cl_device_id> deviceIds(deviceCount);
    checkOpenCLError(clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, deviceCount, deviceIds.data(), nullptr));

    std::vector<OpenCLDevice> devices;
    for (const auto deviceId : deviceIds)
    {
        std::string name = getDeviceInfo(deviceId, CL_DEVICE_NAME);
        devices.push_back(OpenCLDevice(deviceId, name));
    }

    return devices;
}

void OpenCLCore::printOpenCLInfo(std::ostream& outputTarget)
{
    auto platforms = getOpenCLPlatforms();

    for (size_t i = 0; i < platforms.size(); i++)
    {
        outputTarget << "Platform " << i << ": " << platforms.at(i).getName() << std::endl;
        auto devices = getOpenCLDevices(platforms.at(i));

        outputTarget << "Devices for platform " << i << ":" << std::endl;
        for (size_t j = 0; j < devices.size(); j++)
        {
            outputTarget << "Device " << j << ": " << devices.at(j).getName() << std::endl;
        }
        outputTarget << std::endl;
    }
}

void OpenCLCore::createProgram(const std::string& source)
{
    programs.push_back(OpenCLProgram(source, context->getContext(), context->getDevices()));
}

void OpenCLCore::buildProgram(OpenCLProgram& program)
{
    cl_int result = clBuildProgram(program.getProgram(), program.getDevices().size(), program.getDevices().data(), &compilerOptions[0], nullptr,
        nullptr);
    std::string buildInfo = getProgramBuildInfo(program.getProgram(), program.getDevices().at(0));
    checkOpenCLError(result, buildInfo);
}

void OpenCLCore::setOpenCLCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void OpenCLCore::createBuffer(const ArgumentMemoryType& argumentMemoryType, const size_t size)
{
    buffers.push_back(OpenCLBuffer(context->getContext(), getOpenCLMemoryType(argumentMemoryType), size));
}

void OpenCLCore::updateBuffer(OpenCLBuffer& buffer, const void* data, const size_t dataSize)
{
    cl_int result = clEnqueueWriteBuffer(commandQueues.at(0)->getQueue(), buffer.getBuffer(), CL_TRUE, 0, dataSize, data, 0, nullptr, nullptr);
    checkOpenCLError(result);
}

void OpenCLCore::getBufferData(const OpenCLBuffer& buffer, void* data, const size_t dataSize)
{
    cl_int result = clEnqueueReadBuffer(commandQueues.at(0)->getQueue(), buffer.getBuffer(), CL_TRUE, 0, dataSize, data, 0, nullptr, nullptr);
    checkOpenCLError(result);
}

void OpenCLCore::createKernel(const OpenCLProgram& program, const std::string& kernelName)
{
    kernels.push_back(OpenCLKernel(program.getProgram(), kernelName));
}

void OpenCLCore::setKernelArgument(OpenCLKernel& kernel, const OpenCLBuffer& buffer)
{
    kernel.setKernelArgument(buffer.getBuffer(), buffer.getSize());
}

std::string OpenCLCore::getPlatformInfo(const cl_platform_id id, const cl_platform_info info)
{
    size_t infoSize;
    checkOpenCLError(clGetPlatformInfo(id, info, 0, nullptr, &infoSize));
    std::string infoString(infoSize, ' ');
    checkOpenCLError(clGetPlatformInfo(id, info, infoSize, &infoString[0], nullptr));
    
    return infoString;
}

std::string OpenCLCore::getDeviceInfo(const cl_device_id id, const cl_device_info info)
{
    size_t infoSize;
    checkOpenCLError(clGetDeviceInfo(id, info, 0, nullptr, &infoSize));
    std::string infoString(infoSize, ' ');
    checkOpenCLError(clGetDeviceInfo(id, info, infoSize, &infoString[0], nullptr));

    return infoString;
}

std::string OpenCLCore::getProgramBuildInfo(const cl_program program, const cl_device_id id) const
{
    size_t infoSize;
    checkOpenCLError(clGetProgramBuildInfo(program, id, 0, 0, nullptr, &infoSize));
    std::string infoString(infoSize, ' ');
    checkOpenCLError(clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, infoSize, &infoString[0], nullptr));

    return infoString;
}

} // namespace ktt
