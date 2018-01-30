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
    globalSizeCorrection(false),
    programCacheFlag(false),
    nextEventId(0)
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
    EventId eventId = runKernelAsync(kernelData, argumentPointers, getDefaultQueue());
    KernelResult result = getKernelResult(eventId, outputDescriptors);
    return result;
}

EventId OpenclCore::runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue)
{
    Timer overheadTimer;
    overheadTimer.start();

    std::unique_ptr<OpenclProgram> program;
    OpenclProgram* programPointer;

    if (programCacheFlag)
    {
        if (programCache.find(kernelData.getSource()) == programCache.end())
        {
            program = createAndBuildProgram(kernelData.getSource());
            programCache.insert(std::make_pair(kernelData.getSource(), std::move(program)));
        }
        auto cachePointer = programCache.find(kernelData.getSource());
        programPointer = cachePointer->second.get();
    }
    else
    {
        program = createAndBuildProgram(kernelData.getSource());
        programPointer = program.get();
    }
    auto kernel = std::make_unique<OpenclKernel>(programPointer->getProgram(), kernelData.getName());

    for (const auto argument : argumentPointers)
    {
        setKernelArgument(*kernel, *argument);
    }

    overheadTimer.stop();

    return enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), queue, overheadTimer.getElapsedTime());
}

KernelResult OpenclCore::getKernelResult(const EventId id, const std::vector<ArgumentOutputDescriptor>& outputDescriptors) const
{
    auto eventPointer = kernelEvents.find(id);

    if (eventPointer == kernelEvents.end())
    {
        throw std::runtime_error(std::string("Kernel event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    checkOpenclError(clWaitForEvents(1, eventPointer->second->getEvent()), "clWaitForEvents");
    std::string name = eventPointer->second->getKernelName();
    cl_ulong duration = eventPointer->second->getEventCommandDuration();
    uint64_t overhead = eventPointer->second->getOverhead();
    kernelEvents.erase(id);

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

    KernelResult result(name, static_cast<uint64_t>(duration));
    result.setOverhead(overhead);
    return result;
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

void OpenclCore::setProgramCache(const bool flag)
{
    if (!flag)
    {
        clearProgramCache();
    }
    programCacheFlag = flag;
}

void OpenclCore::clearProgramCache()
{
    programCache.clear();
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

void OpenclCore::clearEvents()
{
    kernelEvents.clear();
    bufferEvents.clear();
}

uint64_t OpenclCore::uploadArgument(KernelArgument& kernelArgument)
{
    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return 0;
    }

    EventId eventId = uploadArgumentAsync(kernelArgument, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId OpenclCore::uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return UINT64_MAX;
    }

    clearBuffer(kernelArgument.getId());
    std::unique_ptr<OpenclBuffer> buffer = nullptr;
    EventId eventId = nextEventId;

    if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
    {
        buffer = std::make_unique<OpenclBuffer>(context->getContext(), kernelArgument, true);
        bufferEvents.insert(std::make_pair(eventId, std::make_unique<OpenclEvent>(eventId, false)));
    }
    else
    {
        buffer = std::make_unique<OpenclBuffer>(context->getContext(), kernelArgument, false);
        auto profilingEvent = std::make_unique<OpenclEvent>(eventId, true);
        buffer->uploadData(commandQueues.at(queue)->getQueue(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(),
            profilingEvent->getEvent());
        bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    }

    buffers.insert(std::move(buffer)); // buffer data will be stolen
    nextEventId++;
    return eventId;
}

uint64_t OpenclCore::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes)
{
    EventId eventId = updateArgumentAsync(id, data, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId OpenclCore::updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    OpenclBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenclEvent>(eventId, true);
    buffer->uploadData(commandQueues.at(queue)->getQueue(), data, dataSizeInBytes, profilingEvent->getEvent());

    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    nextEventId++;
    return eventId;
}

uint64_t OpenclCore::downloadArgument(const ArgumentId id, void* destination) const
{
    EventId eventId = downloadArgumentAsync(id, destination, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId OpenclCore::downloadArgumentAsync(const ArgumentId id, void* destination, const QueueId queue) const
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    OpenclBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenclEvent>(eventId, true);
    buffer->downloadData(commandQueues.at(queue)->getQueue(), destination, buffer->getBufferSize(), profilingEvent->getEvent());

    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    nextEventId++;
    return eventId;
}

uint64_t OpenclCore::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const
{
    EventId eventId = downloadArgumentAsync(id, destination, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId OpenclCore::downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    OpenclBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenclEvent>(eventId, true);
    buffer->downloadData(commandQueues.at(queue)->getQueue(), destination, dataSizeInBytes, profilingEvent->getEvent());

    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    nextEventId++;
    return eventId;
}

KernelArgument OpenclCore::downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const
{
    OpenclBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getElementSize(),
        buffer->getDataType(), buffer->getMemoryLocation(), buffer->getAccessType(), ArgumentUploadType::Vector);

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenclEvent>(eventId, true);
    buffer->downloadData(commandQueues.at(getDefaultQueue())->getQueue(), argument.getData(), argument.getDataSizeInBytes(),
        profilingEvent->getEvent());

    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    nextEventId++;

    uint64_t duration = getArgumentOperationDuration(eventId);
    if (downloadDuration != nullptr)
    {
        *downloadDuration = duration;
    }
    
    return argument;
}

uint64_t OpenclCore::getArgumentOperationDuration(const EventId id) const
{
    auto eventPointer = bufferEvents.find(id);

    if (eventPointer == bufferEvents.end())
    {
        throw std::runtime_error(std::string("Buffer event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    if (!eventPointer->second->isValid())
    {
        bufferEvents.erase(id);
        return 0;
    }

    checkOpenclError(clWaitForEvents(1, eventPointer->second->getEvent()), "clWaitForEvents");
    cl_ulong duration = eventPointer->second->getEventCommandDuration();
    bufferEvents.erase(id);

    return static_cast<uint64_t>(duration);
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

EventId OpenclCore::enqueueKernel(OpenclKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    const QueueId queue, const uint64_t kernelLaunchOverhead) const
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
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

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenclEvent>(eventId, kernel.getKernelName(), kernelLaunchOverhead);
    nextEventId++;

    cl_int result = clEnqueueNDRangeKernel(commandQueues.at(queue)->getQueue(), kernel.getKernel(),
        static_cast<cl_uint>(correctedGlobalSize.size()), nullptr, correctedGlobalSize.data(), localSize.data(), 0, nullptr, profilingEvent->getEvent());
    checkOpenclError(result, "clEnqueueNDRangeKernel");

    kernelEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    return eventId;
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
