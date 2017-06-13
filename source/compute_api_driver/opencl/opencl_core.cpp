#include "opencl_core.h"
#include "utility/timer.h"

namespace ktt
{

OpenclCore::OpenclCore(const size_t platformIndex, const size_t deviceIndex) :
    platformIndex(platformIndex),
    deviceIndex(deviceIndex),
    compilerOptions(std::string(""))
{
    auto platforms = getOpenclPlatforms();
    if (platformIndex >= platforms.size())
    {
        throw std::runtime_error(std::string("Invalid platform index: ") + std::to_string(platformIndex));
    }

    auto devices = getOpenclDevices(platforms.at(platformIndex));
    if (deviceIndex >= devices.size())
    {
        throw std::runtime_error(std::string("Invalid device index: ") + std::to_string(deviceIndex));
    }

    cl_device_id device = devices.at(deviceIndex).getId();
    context = std::make_unique<OpenclContext>(platforms.at(platformIndex).getId(), std::vector<cl_device_id>{ device });
    commandQueue = std::make_unique<OpenclCommandQueue>(context->getContext(), device);
}

void OpenclCore::printComputeApiInfo(std::ostream& outputTarget) const
{
    auto platforms = getOpenclPlatforms();

    for (size_t i = 0; i < platforms.size(); i++)
    {
        outputTarget << "Platform " << i << ": " << platforms.at(i).getName() << std::endl;
        auto devices = getOpenclDevices(platforms.at(i));

        outputTarget << "Devices for platform " << i << ":" << std::endl;
        for (size_t j = 0; j < devices.size(); j++)
        {
            outputTarget << "Device " << j << ": " << devices.at(j).getName() << std::endl;
        }
        outputTarget << std::endl;
    }
}

std::vector<PlatformInfo> OpenclCore::getPlatformInfo() const
{
    std::vector<PlatformInfo> result;
    auto platforms = getOpenclPlatforms();

    for (size_t i = 0; i < platforms.size(); i++)
    {
        result.push_back(getOpenclPlatformInfo(i));
    }

    return result;
}

std::vector<DeviceInfo> OpenclCore::getDeviceInfo(const size_t platformIndex) const
{
    std::vector<DeviceInfo> result;
    auto platforms = getOpenclPlatforms();
    auto devices = getOpenclDevices(platforms.at(platformIndex));

    for (size_t i = 0; i < devices.size(); i++)
    {
        result.push_back(getOpenclDeviceInfo(platformIndex, i));
    }

    return result;
}

DeviceInfo OpenclCore::getCurrentDeviceInfo() const
{
    return getOpenclDeviceInfo(platformIndex, deviceIndex);
}

void OpenclCore::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void OpenclCore::uploadArgument(const KernelArgument& kernelArgument)
{
    if (kernelArgument.getArgumentUploadType() != ArgumentUploadType::Vector)
    {
        return;
    }
    
    clearBuffer(kernelArgument.getId());

    std::unique_ptr<OpenclBuffer> buffer = createBuffer(kernelArgument);
    buffer->uploadData(*commandQueue, kernelArgument.getData(), kernelArgument.getDataSizeInBytes());
    buffers.insert(std::move(buffer)); // buffer data will be stolen
}

void OpenclCore::updateArgument(const size_t argumentId, const void* data, const size_t dataSizeInBytes)
{
    for (const auto& buffer : buffers)
    {
        if (buffer->getKernelArgumentId() == argumentId)
        {
            buffer->uploadData(*commandQueue, data, dataSizeInBytes);
            return;
        }
    }
    throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(argumentId));
}

KernelArgument OpenclCore::downloadArgument(const size_t argumentId) const
{
    for (const auto& buffer : buffers)
    {
        if (buffer->getKernelArgumentId() != argumentId)
        {
            continue;
        }

        KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getDataType(),
            buffer->getMemoryType(), ArgumentUploadType::Vector);
        buffer->downloadData(*commandQueue, argument.getData(), argument.getDataSizeInBytes());
        return argument;
    }

    throw std::runtime_error(std::string("Invalid argument id: ") + std::to_string(argumentId));
}

void OpenclCore::clearBuffer(const size_t argumentId)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getKernelArgumentId() == argumentId)
        {
            buffers.erase(iterator);
            return;
        }
        else
        {
            ++iterator;
        }
    }
}

void OpenclCore::clearBuffers()
{
    buffers.clear();
}

void OpenclCore::clearBuffers(const ArgumentMemoryType& argumentMemoryType)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getOpenclMemoryFlag() == getOpenclMemoryType(argumentMemoryType))
        {
            iterator = buffers.erase(iterator);
        }
        else
        {
            ++iterator;
        }
    }
}

KernelRunResult OpenclCore::runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
    const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers)
{
    Timer timer;
    timer.start();

    std::unique_ptr<OpenclProgram> program = createAndBuildProgram(source);
    std::unique_ptr<OpenclKernel> kernel = createKernel(*program, kernelName);

    for (const auto argument : argumentPointers)
    {
        setKernelArgument(*kernel, *argument);
    }

    cl_ulong duration = enqueueKernel(*kernel, globalSize, localSize);

    timer.stop();
    uint64_t overhead = timer.getElapsedTime();
    return KernelRunResult(static_cast<uint64_t>(duration), overhead);
}

std::unique_ptr<OpenclProgram> OpenclCore::createAndBuildProgram(const std::string& source) const
{
    auto program = std::make_unique<OpenclProgram>(source, context->getContext(), context->getDevices());
    program->build(compilerOptions);
    return program;
}

std::unique_ptr<OpenclBuffer> OpenclCore::createBuffer(const KernelArgument& argument) const
{
    auto buffer = std::make_unique<OpenclBuffer>(context->getContext(), argument.getId(), argument.getDataSizeInBytes(),
        argument.getElementSizeInBytes(), argument.getArgumentDataType(), argument.getArgumentMemoryType());
    return buffer;
}

void OpenclCore::setKernelArgument(OpenclKernel& kernel, const KernelArgument& argument)
{
    if (argument.getArgumentUploadType() == ArgumentUploadType::Vector)
    {
        if (!loadBufferFromCache(argument.getId(), kernel))
        {
            uploadArgument(argument);
            loadBufferFromCache(argument.getId(), kernel);
        }
    }
    else if(argument.getArgumentUploadType() == ArgumentUploadType::Scalar)
    {
        kernel.setKernelArgumentScalar(argument.getData(), argument.getElementSizeInBytes());
    }
    else
    {
        kernel.setKernelArgumentLocal(argument.getElementSizeInBytes() * argument.getNumberOfElements());
    }
}

std::unique_ptr<OpenclKernel> OpenclCore::createKernel(const OpenclProgram& program, const std::string& kernelName) const
{
    auto kernel = std::make_unique<OpenclKernel>(program.getProgram(), kernelName);
    return kernel;
}

cl_ulong OpenclCore::enqueueKernel(OpenclKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize) const
{
    cl_event profilingEvent;
    cl_int result = clEnqueueNDRangeKernel(commandQueue->getQueue(), kernel.getKernel(), static_cast<cl_uint>(globalSize.size()), nullptr,
        globalSize.data(), localSize.data(), 0, nullptr, &profilingEvent);
    checkOpenclError(result, std::string("clEnqueueNDRangeKernel"));

    // Wait for computation to finish
    checkOpenclError(clWaitForEvents(1, &profilingEvent), std::string("clWaitForEvents"));

    return getKernelRunDuration(profilingEvent);
}

PlatformInfo OpenclCore::getOpenclPlatformInfo(const size_t platformIndex)
{
    auto platforms = getOpenclPlatforms();
    PlatformInfo result(platformIndex, platforms.at(platformIndex).getName());

    cl_platform_id id = platforms.at(platformIndex).getId();
    result.setExtensions(getPlatformInfoString(id, CL_PLATFORM_EXTENSIONS));
    result.setVendor(getPlatformInfoString(id, CL_PLATFORM_VENDOR));
    result.setVersion(getPlatformInfoString(id, CL_PLATFORM_VERSION));

    return result;
}

DeviceInfo OpenclCore::getOpenclDeviceInfo(const size_t platformIndex, const size_t deviceIndex)
{
    auto platforms = getOpenclPlatforms();
    auto devices = getOpenclDevices(platforms.at(platformIndex));
    DeviceInfo result(deviceIndex, devices.at(deviceIndex).getName());

    cl_device_id id = devices.at(deviceIndex).getId();
    result.setExtensions(getDeviceInfoString(id, CL_DEVICE_EXTENSIONS));
    result.setVendor(getDeviceInfoString(id, CL_DEVICE_VENDOR));
        
    uint64_t globalMemorySize;
    checkOpenclError(clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(uint64_t), &globalMemorySize, nullptr));
    result.setGlobalMemorySize(globalMemorySize);

    uint64_t localMemorySize;
    checkOpenclError(clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(uint64_t), &localMemorySize, nullptr));
    result.setLocalMemorySize(localMemorySize);

    uint64_t maxConstantBufferSize;
    checkOpenclError(clGetDeviceInfo(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(uint64_t), &maxConstantBufferSize, nullptr));
    result.setMaxConstantBufferSize(maxConstantBufferSize);

    uint32_t maxComputeUnits;
    checkOpenclError(clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint32_t), &maxComputeUnits, nullptr));
    result.setMaxComputeUnits(maxComputeUnits);

    size_t maxWorkGroupSize;
    checkOpenclError(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr));
    result.setMaxWorkGroupSize(maxWorkGroupSize);

    cl_device_type deviceType;
    checkOpenclError(clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
    result.setDeviceType(getDeviceType(deviceType));

    return result;
}

std::vector<OpenclPlatform> OpenclCore::getOpenclPlatforms()
{
    cl_uint platformCount;
    checkOpenclError(clGetPlatformIDs(0, nullptr, &platformCount));

    std::vector<cl_platform_id> platformIds(platformCount);
    checkOpenclError(clGetPlatformIDs(platformCount, platformIds.data(), nullptr));

    std::vector<OpenclPlatform> platforms;
    for (const auto platformId : platformIds)
    {
        std::string name = getPlatformInfoString(platformId, CL_PLATFORM_NAME);
        platforms.push_back(OpenclPlatform(platformId, name));
    }

    return platforms;
}

std::vector<OpenclDevice> OpenclCore::getOpenclDevices(const OpenclPlatform& platform)
{
    cl_uint deviceCount;
    checkOpenclError(clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount));

    std::vector<cl_device_id> deviceIds(deviceCount);
    checkOpenclError(clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, deviceCount, deviceIds.data(), nullptr));

    std::vector<OpenclDevice> devices;
    for (const auto deviceId : deviceIds)
    {
        std::string name = getDeviceInfoString(deviceId, CL_DEVICE_NAME);
        devices.push_back(OpenclDevice(deviceId, name));
    }

    return devices;
}

DeviceType OpenclCore::getDeviceType(const cl_device_type deviceType)
{
    switch (deviceType)
    {
    case CL_DEVICE_TYPE_CPU:
        return DeviceType::CPU;
    case CL_DEVICE_TYPE_GPU:
        return DeviceType::GPU;
    case CL_DEVICE_TYPE_ACCELERATOR:
        return DeviceType::Accelerator;
    case CL_DEVICE_TYPE_DEFAULT:
        return DeviceType::Default;
    default:
        return DeviceType::Custom;
    }
}

void OpenclCore::setKernelArgumentVector(OpenclKernel& kernel, const OpenclBuffer& buffer) const
{
    cl_mem clBuffer = buffer.getBuffer();
    kernel.setKernelArgumentVector((void*)&clBuffer);
}

bool OpenclCore::loadBufferFromCache(const size_t argumentId, OpenclKernel& kernel) const
{
    for (const auto& buffer : buffers)
    {
        if (buffer->getKernelArgumentId() == argumentId)
        {
            setKernelArgumentVector(kernel, *buffer);
            return true;
        }
    }
    return false;
}

} // namespace ktt
