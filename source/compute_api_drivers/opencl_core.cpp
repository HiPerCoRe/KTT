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

std::vector<Platform> OpenCLCore::getOpenCLPlatformInfo()
{
    auto platforms = getOpenCLPlatforms();

    std::vector<Platform> result;
    for (size_t i = 0; i < platforms.size(); i++)
    {
        Platform current(i, platforms.at(i).getName());
        cl_platform_id currentId = platforms.at(i).getId();
        current.setExtensions(getPlatformInfo(currentId, CL_PLATFORM_EXTENSIONS));
        current.setVendor(getPlatformInfo(currentId, CL_PLATFORM_VENDOR));
        current.setVersion(getPlatformInfo(currentId, CL_PLATFORM_VERSION));

        result.push_back(current);
    }

    return result;
}

std::vector<Device> OpenCLCore::getOpenCLDeviceInfo(const size_t platformIndex)
{
    auto platforms = getOpenCLPlatforms();
    auto devices = getOpenCLDevices(platforms.at(platformIndex));

    std::vector<Device> result;
    for (size_t i = 0; i < devices.size(); i++)
    {
        Device current(i, devices.at(i).getName());
        cl_device_id currentId = devices.at(i).getId();
        current.setExtensions(getDeviceInfo(currentId, CL_DEVICE_EXTENSIONS));
        current.setVendor(getDeviceInfo(currentId, CL_DEVICE_VENDOR));
        
        uint64_t globalMemorySize;
        checkOpenCLError(clGetDeviceInfo(currentId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(uint64_t), &globalMemorySize, nullptr));
        current.setGlobalMemorySize(globalMemorySize);

        uint64_t localMemorySize;
        checkOpenCLError(clGetDeviceInfo(currentId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(uint64_t), &localMemorySize, nullptr));
        current.setLocalMemorySize(localMemorySize);

        uint64_t maxConstantBufferSize;
        checkOpenCLError(clGetDeviceInfo(currentId, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(uint64_t), &maxConstantBufferSize, nullptr));
        current.setMaxConstantBufferSize(maxConstantBufferSize);

        uint32_t maxComputeUnits;
        checkOpenCLError(clGetDeviceInfo(currentId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint32_t), &maxComputeUnits, nullptr));
        current.setMaxComputeUnits(maxComputeUnits);

        size_t maxWorkGroupSize;
        checkOpenCLError(clGetDeviceInfo(currentId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr));
        current.setMaxWorkGroupSize(maxWorkGroupSize);

        cl_device_type deviceType;
        checkOpenCLError(clGetDeviceInfo(currentId, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
        current.setDeviceType(getDeviceType(deviceType));

        result.push_back(current);
    }

    return result;
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

DeviceType OpenCLCore::getDeviceType(const cl_device_type deviceType)
{
    switch (deviceType)
    {
    case CL_DEVICE_TYPE_CPU:
        return DeviceType::CPU;
    case CL_DEVICE_TYPE_GPU:
        return DeviceType::GPU;
    case CL_DEVICE_TYPE_ACCELERATOR:
        return DeviceType::ACCELERATOR;
    case CL_DEVICE_TYPE_DEFAULT:
        return DeviceType::DEFAULT;
    default:
        return DeviceType::CUSTOM;
    }
}

} // namespace ktt
