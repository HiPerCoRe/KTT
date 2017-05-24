#include "opencl_core.h"
#include "../../utility/timer.h"

namespace ktt
{

OpenclCore::OpenclCore(const size_t platformIndex, const size_t deviceIndex) :
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

void OpenclCore::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void OpenclCore::clearCache() const
{
    bufferCache.clear();
}

KernelRunResult OpenclCore::runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
    const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers) const
{
    Timer timer;
    timer.start();

    std::unique_ptr<OpenclProgram> program = createAndBuildProgram(source);
    std::unique_ptr<OpenclKernel> kernel = createKernel(*program, kernelName);
    std::vector<std::unique_ptr<OpenclBuffer>> buffers;
    std::vector<const KernelArgument*> vectorArgumentPointers;

    for (const auto& argument : argumentPointers)
    {
        if (argument->getArgumentUploadType() == ArgumentUploadType::Vector)
        {
            if (argument->getArgumentMemoryType() == ArgumentMemoryType::ReadOnly && loadBufferFromCache(argument->getId(), *kernel))
            {
                continue; // buffer was successfully loaded from cache
            }

            std::unique_ptr<OpenclBuffer> buffer = createBuffer(argument->getArgumentMemoryType(), argument->getDataSizeInBytes(),
                argument->getId());
            updateBuffer(*buffer, argument->getData(), argument->getDataSizeInBytes());
            setKernelArgumentVector(*kernel, *buffer);

            if (argument->getArgumentMemoryType() == ArgumentMemoryType::ReadOnly)
            {
                bufferCache.push_back(std::move(buffer));
            }
            else
            {
                vectorArgumentPointers.push_back(argument);
                buffers.push_back(std::move(buffer)); // buffer data will be stolen
            }
        }
        else
        {
            setKernelArgumentScalar(*kernel, *argument);
        }
    }

    cl_ulong duration = enqueueKernel(*kernel, globalSize, localSize);
    std::vector<KernelArgument> resultArguments = getResultArguments(buffers, vectorArgumentPointers);

    timer.stop();
    uint64_t overhead = timer.getElapsedTime();
    return KernelRunResult(static_cast<uint64_t>(duration), overhead, resultArguments);
}

std::unique_ptr<OpenclProgram> OpenclCore::createAndBuildProgram(const std::string& source) const
{
    std::unique_ptr<OpenclProgram> program;
    program.reset(new OpenclProgram(source, context->getContext(), context->getDevices()));
    program->build(compilerOptions);
    return program;
}

std::unique_ptr<OpenclBuffer> OpenclCore::createBuffer(const ArgumentMemoryType& argumentMemoryType, const size_t size,
    const size_t kernelArgumentId) const
{
    std::unique_ptr<OpenclBuffer> buffer;
    buffer.reset(new OpenclBuffer(context->getContext(), getOpenclMemoryType(argumentMemoryType), size, kernelArgumentId));
    return buffer;
}

void OpenclCore::updateBuffer(OpenclBuffer& buffer, const void* source, const size_t dataSize) const
{
    cl_int result = clEnqueueWriteBuffer(commandQueue->getQueue(), buffer.getBuffer(), CL_TRUE, 0, dataSize, source, 0, nullptr, nullptr);
    checkOpenclError(result, std::string("clEnqueueWriteBuffer"));
}

void OpenclCore::getBufferData(const OpenclBuffer& buffer, void* destination, const size_t dataSize) const
{
    cl_int result = clEnqueueReadBuffer(commandQueue->getQueue(), buffer.getBuffer(), CL_TRUE, 0, dataSize, destination, 0, nullptr, nullptr);
    checkOpenclError(result, std::string("clEnqueueReadBuffer"));
}

std::unique_ptr<OpenclKernel> OpenclCore::createKernel(const OpenclProgram& program, const std::string& kernelName) const
{
    std::unique_ptr<OpenclKernel> kernel;
    kernel.reset(new OpenclKernel(program.getProgram(), kernelName));
    return kernel;
}

void OpenclCore::setKernelArgumentScalar(OpenclKernel& kernel, const KernelArgument& argument) const
{
    kernel.setKernelArgumentScalar(argument.getData(), argument.getElementSizeInBytes());
}

void OpenclCore::setKernelArgumentVector(OpenclKernel& kernel, const OpenclBuffer& buffer) const
{
    cl_mem clBuffer = buffer.getBuffer();
    kernel.setKernelArgumentVector((void*)&clBuffer);
}

cl_ulong OpenclCore::enqueueKernel(OpenclKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize) const
{
    cl_event profilingEvent;
    cl_int result = clEnqueueNDRangeKernel(commandQueue->getQueue(), kernel.getKernel(), static_cast<cl_uint>(globalSize.size()), nullptr,
        globalSize.data(), localSize.data(), 0, nullptr, &profilingEvent);
    checkOpenclError(result, std::string("clEnqueueNDRangeKernel"));

    clFinish(commandQueue->getQueue());
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

std::vector<KernelArgument> OpenclCore::getResultArguments(const std::vector<std::unique_ptr<OpenclBuffer>>& outputBuffers,
    const std::vector<const KernelArgument*>& inputArgumentPointers) const
{
    std::vector<KernelArgument> resultArguments;
    for (size_t i = 0; i < outputBuffers.size(); i++)
    {
        if (outputBuffers.at(i)->getType() == CL_MEM_READ_ONLY)
        {
            continue;
        }

        const KernelArgument* currentArgument = inputArgumentPointers.at(i);
        KernelArgument argument(currentArgument->getId(), currentArgument->getNumberOfElements(), currentArgument->getArgumentDataType(),
            currentArgument->getArgumentMemoryType(), currentArgument->getArgumentUploadType());
        getBufferData(*outputBuffers.at(i), argument.getData(), argument.getDataSizeInBytes());
        resultArguments.push_back(argument);
    }

    return resultArguments;
}

bool OpenclCore::loadBufferFromCache(const size_t argumentId, OpenclKernel& kernel) const
{
    for (const auto& buffer : bufferCache)
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
