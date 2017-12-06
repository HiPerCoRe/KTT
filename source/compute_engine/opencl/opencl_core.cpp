#include "opencl_core.h"
#include "utility/ktt_utility.h"
#include "utility/timer.h"

namespace ktt
{

OpenclCore::OpenclCore(const size_t platformIndex, const size_t deviceIndex, const size_t queueCount) :
    platformIndex(platformIndex),
    deviceIndex(deviceIndex),
    queueCount(queueCount),
    compilerOptions(std::string("")),
    globalSizeType(GlobalSizeType::Opencl),
    globalSizeCorrection(false)
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
    for (size_t i = 0; i < queueCount; i++)
    {
        auto commandQueue = std::make_unique<OpenclCommandQueue>(i, context->getContext(), device);
        commandQueues.push_back(std::move(commandQueue));
    }
}

KernelResult OpenclCore::runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
    const std::vector<ArgumentOutputDescriptor>& outputDescriptors)
{
    std::unique_ptr<OpenclProgram> program = createAndBuildProgram(kernelData.getSource());
    auto kernel = std::make_unique<OpenclKernel>(program->getProgram(), kernelData.getName());

    for (const auto argument : argumentPointers)
    {
        setKernelArgument(*kernel, *argument);
    }

    cl_ulong duration = enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), getDefaultQueue(), true);

    for (const auto& descriptor : outputDescriptors)
    {
        if (descriptor.getOutputSizeInBytes() == 0)
        {
            downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination());
        }
        else
        {
            downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
        }
    }

    return KernelResult(kernelData.getName(), static_cast<uint64_t>(duration));
}

void OpenclCore::runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue,
    const bool synchronizeFlag)
{
    std::unique_ptr<OpenclProgram> program = createAndBuildProgram(kernelData.getSource());
    auto kernel = std::make_unique<OpenclKernel>(program->getProgram(), kernelData.getName());

    for (const auto argument : argumentPointers)
    {
        setKernelArgument(*kernel, *argument);
    }

    enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), queue, synchronizeFlag);
}

void OpenclCore::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void OpenclCore::setGlobalSizeType(const GlobalSizeType& type)
{
    globalSizeType = type;
}

void OpenclCore::setAutomaticGlobalSizeCorrection(const bool flag)
{
    globalSizeCorrection = flag;
}

QueueId OpenclCore::getDefaultQueue() const
{
    return 0;
}

std::vector<QueueId> OpenclCore::getAllQueues() const
{
    std::vector<QueueId> result;

    for (size_t i = 0; i < commandQueues.size(); i++)
    {
        result.push_back(i);
    }

    return result;
}

void OpenclCore::synchronizeQueue(const QueueId queue)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid command queue index: ") + std::to_string(queue));
    }

    checkOpenclError(clFinish(commandQueues.at(queue)->getQueue()), "clFinish");
}

void OpenclCore::synchronizeDevice()
{
    for (auto& commandQueue : commandQueues)
    {
        checkOpenclError(clFinish(commandQueue->getQueue()), "clFinish");
    }
}

void OpenclCore::uploadArgument(KernelArgument& kernelArgument)
{
    uploadArgument(kernelArgument, getDefaultQueue(), true);
}

void OpenclCore::uploadArgument(KernelArgument& kernelArgument, const QueueId queue, const bool synchronizeFlag)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return;
    }

    clearBuffer(kernelArgument.getId());
    std::unique_ptr<OpenclBuffer> buffer = nullptr;

    if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
    {
        buffer = std::make_unique<OpenclBuffer>(context->getContext(), kernelArgument, true);
    }
    else
    {
        buffer = std::make_unique<OpenclBuffer>(context->getContext(), kernelArgument, false);
        buffer->uploadData(commandQueues.at(queue)->getQueue(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(), synchronizeFlag);
    }

    buffers.insert(std::move(buffer)); // buffer data will be stolen
}

void OpenclCore::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes)
{
    updateArgument(id, data, dataSizeInBytes, getDefaultQueue(), true);
}

void OpenclCore::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue,
    const bool synchronizeFlag)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    OpenclBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    buffer->uploadData(commandQueues.at(queue)->getQueue(), data, dataSizeInBytes, synchronizeFlag);
}

void OpenclCore::downloadArgument(const ArgumentId id, void* destination) const
{
    downloadArgument(id, destination, getDefaultQueue(), true);
}

void OpenclCore::downloadArgument(const ArgumentId id, void* destination, const QueueId queue, const bool synchronizeFlag) const
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    OpenclBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    buffer->downloadData(commandQueues.at(queue)->getQueue(), destination, buffer->getBufferSize(), synchronizeFlag);
}

void OpenclCore::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const
{
    downloadArgument(id, destination, dataSizeInBytes, getDefaultQueue(), true);
}

void OpenclCore::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue,
    const bool synchronizeFlag) const
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    OpenclBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    buffer->downloadData(commandQueues.at(queue)->getQueue(), destination, dataSizeInBytes, synchronizeFlag);
}

KernelArgument OpenclCore::downloadArgument(const ArgumentId id) const
{
    OpenclBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getElementSize(),
        buffer->getDataType(), buffer->getMemoryLocation(), buffer->getAccessType(), ArgumentUploadType::Vector);
    buffer->downloadData(commandQueues.at(getDefaultQueue())->getQueue(), argument.getData(), argument.getDataSizeInBytes(), true);
    
    return argument;
}

void OpenclCore::clearBuffer(const ArgumentId id)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getKernelArgumentId() == id)
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

void OpenclCore::clearBuffers(const ArgumentAccessType& accessType)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getOpenclMemoryFlag() == getOpenclMemoryType(accessType))
        {
            iterator = buffers.erase(iterator);
        }
        else
        {
            ++iterator;
        }
    }
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

std::unique_ptr<OpenclProgram> OpenclCore::createAndBuildProgram(const std::string& source) const
{
    auto program = std::make_unique<OpenclProgram>(source, context->getContext(), context->getDevices());
    program->build(compilerOptions);
    return program;
}

void OpenclCore::setKernelArgument(OpenclKernel& kernel, KernelArgument& argument)
{
    if (argument.getUploadType() == ArgumentUploadType::Vector)
    {
        if (!loadBufferFromCache(argument.getId(), kernel))
        {
            uploadArgument(argument);
            loadBufferFromCache(argument.getId(), kernel);
        }
    }
    else if (argument.getUploadType() == ArgumentUploadType::Scalar)
    {
        kernel.setKernelArgumentScalar(argument.getData(), argument.getElementSizeInBytes());
    }
    else
    {
        kernel.setKernelArgumentLocal(argument.getElementSizeInBytes() * argument.getNumberOfElements());
    }
}

cl_ulong OpenclCore::enqueueKernel(OpenclKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    const QueueId queue, const bool synchronizeFlag) const
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid stream index: ") + std::to_string(queue));
    }

    std::vector<size_t> correctedGlobalSize = globalSize;
    if (globalSizeType == GlobalSizeType::Cuda)
    {
        correctedGlobalSize.at(0) *= localSize.at(0);
        correctedGlobalSize.at(1) *= localSize.at(1);
        correctedGlobalSize.at(2) *= localSize.at(2);
    }
    if (globalSizeCorrection)
    {
        correctedGlobalSize = roundUpGlobalSize(correctedGlobalSize, localSize);
    }

    if (!synchronizeFlag)
    {
        cl_int result = clEnqueueNDRangeKernel(commandQueues.at(queue)->getQueue(), kernel.getKernel(),
            static_cast<cl_uint>(correctedGlobalSize.size()), nullptr, correctedGlobalSize.data(), localSize.data(), 0, nullptr, nullptr);
        checkOpenclError(result, "clEnqueueNDRangeKernel");
        return 0;
    }

    cl_event profilingEvent;
    cl_int result = clEnqueueNDRangeKernel(commandQueues.at(queue)->getQueue(), kernel.getKernel(),
        static_cast<cl_uint>(correctedGlobalSize.size()), nullptr, correctedGlobalSize.data(), localSize.data(), 0, nullptr, &profilingEvent);
    checkOpenclError(result, "clEnqueueNDRangeKernel");

    // Wait for computation to finish
    checkOpenclError(clWaitForEvents(1, &profilingEvent), "clWaitForEvents");

    cl_ulong duration = getKernelRunDuration(profilingEvent);
    checkOpenclError(clReleaseEvent(profilingEvent), "clReleaseEvent");

    return duration;
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

OpenclBuffer* OpenclCore::findBuffer(const ArgumentId id) const
{
    for (const auto& buffer : buffers)
    {
        if (buffer->getKernelArgumentId() == id)
        {
            return buffer.get();
        }
    }

    return nullptr;
}

void OpenclCore::setKernelArgumentVector(OpenclKernel& kernel, const OpenclBuffer& buffer) const
{
    cl_mem clBuffer = buffer.getBuffer();
    kernel.setKernelArgumentVector((void*)&clBuffer);
}

bool OpenclCore::loadBufferFromCache(const ArgumentId id, OpenclKernel& kernel) const
{
    OpenclBuffer* buffer = findBuffer(id);

    if (buffer != nullptr)
    {
        setKernelArgumentVector(kernel, *buffer);
        return true;
    }

    return false;
}

} // namespace ktt
