#include "opencl_engine.h"
#include "utility/ktt_utility.h"
#include "utility/timer.h"

namespace ktt
{

#ifdef PLATFORM_OPENCL

OpenCLEngine::OpenCLEngine(const PlatformIndex platformIndex, const DeviceIndex deviceIndex, const uint32_t queueCount) :
    platformIndex(platformIndex),
    deviceIndex(deviceIndex),
    queueCount(queueCount),
    compilerOptions(std::string("")),
    globalSizeType(GlobalSizeType::OpenCL),
    globalSizeCorrection(false),
    programCacheFlag(false),
    nextEventId(0)
{
    auto platforms = getOpenCLPlatforms();
    if (platformIndex >= platforms.size())
    {
        throw std::runtime_error(std::string("Invalid platform index: ") + std::to_string(platformIndex));
    }

    auto devices = getOpenCLDevices(platforms.at(platformIndex));
    if (deviceIndex >= devices.size())
    {
        throw std::runtime_error(std::string("Invalid device index: ") + std::to_string(deviceIndex));
    }

    cl_device_id device = devices.at(deviceIndex).getId();
    context = std::make_unique<OpenCLContext>(platforms.at(platformIndex).getId(), std::vector<cl_device_id>{device});
    for (uint32_t i = 0; i < queueCount; i++)
    {
        auto commandQueue = std::make_unique<OpenCLCommandQueue>(i, context->getContext(), device);
        commandQueues.push_back(std::move(commandQueue));
    }
}

KernelResult OpenCLEngine::runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
    const std::vector<OutputDescriptor>& outputDescriptors)
{
    EventId eventId = runKernelAsync(kernelData, argumentPointers, getDefaultQueue());
    KernelResult result = getKernelResult(eventId, outputDescriptors);
    return result;
}

EventId OpenCLEngine::runKernelAsync(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers, const QueueId queue)
{
    Timer overheadTimer;
    overheadTimer.start();

    std::unique_ptr<OpenCLProgram> program;
    OpenCLProgram* programPointer;

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
    auto kernel = std::make_unique<OpenCLKernel>(programPointer->getProgram(), kernelData.getName());

    checkLocalMemoryModifiers(argumentPointers, kernelData.getLocalMemoryModifiers());
    for (const auto argument : argumentPointers)
    {
        if (argument->getUploadType() == ArgumentUploadType::Local)
        {
            setKernelArgument(*kernel, *argument, kernelData.getLocalMemoryModifiers());
        }
        else
        {
            setKernelArgument(*kernel, *argument);
        }
    }

    overheadTimer.stop();

    return enqueueKernel(*kernel, kernelData.getGlobalSize(), kernelData.getLocalSize(), queue, overheadTimer.getElapsedTime());
}

KernelResult OpenCLEngine::getKernelResult(const EventId id, const std::vector<OutputDescriptor>& outputDescriptors) const
{
    auto eventPointer = kernelEvents.find(id);

    if (eventPointer == kernelEvents.end())
    {
        throw std::runtime_error(std::string("Kernel event with following id does not exist or its result was already retrieved: ")
            + std::to_string(id));
    }

    checkOpenCLError(clWaitForEvents(1, eventPointer->second->getEvent()), "clWaitForEvents");
    std::string name = eventPointer->second->getKernelName();
    cl_ulong duration = eventPointer->second->getEventCommandDuration();
    uint64_t overhead = eventPointer->second->getOverhead();
    kernelEvents.erase(id);

    for (const auto& descriptor : outputDescriptors)
    {
        downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
    }

    KernelResult result(name, static_cast<uint64_t>(duration));
    result.setOverhead(overhead);
    return result;
}

void OpenCLEngine::setCompilerOptions(const std::string& options)
{
    compilerOptions = options;
}

void OpenCLEngine::setGlobalSizeType(const GlobalSizeType type)
{
    globalSizeType = type;
}

void OpenCLEngine::setAutomaticGlobalSizeCorrection(const bool flag)
{
    globalSizeCorrection = flag;
}

void OpenCLEngine::setProgramCache(const bool flag)
{
    if (!flag)
    {
        clearProgramCache();
    }
    programCacheFlag = flag;
}

void OpenCLEngine::clearProgramCache()
{
    programCache.clear();
}

QueueId OpenCLEngine::getDefaultQueue() const
{
    return 0;
}

std::vector<QueueId> OpenCLEngine::getAllQueues() const
{
    std::vector<QueueId> result;

    for (size_t i = 0; i < commandQueues.size(); i++)
    {
        result.push_back(static_cast<QueueId>(i));
    }

    return result;
}

void OpenCLEngine::synchronizeQueue(const QueueId queue)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid command queue index: ") + std::to_string(queue));
    }

    checkOpenCLError(clFinish(commandQueues.at(queue)->getQueue()), "clFinish");
}

void OpenCLEngine::synchronizeDevice()
{
    for (auto& commandQueue : commandQueues)
    {
        checkOpenCLError(clFinish(commandQueue->getQueue()), "clFinish");
    }
}

void OpenCLEngine::clearEvents()
{
    kernelEvents.clear();
    bufferEvents.clear();
}

uint64_t OpenCLEngine::uploadArgument(KernelArgument& kernelArgument)
{
    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return 0;
    }

    EventId eventId = uploadArgumentAsync(kernelArgument, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId OpenCLEngine::uploadArgumentAsync(KernelArgument& kernelArgument, const QueueId queue)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    if (findBuffer(kernelArgument.getId()) != nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id already exists: ") + std::to_string(kernelArgument.getId()));
    }

    if (kernelArgument.getUploadType() != ArgumentUploadType::Vector)
    {
        return UINT64_MAX;
    }

    std::unique_ptr<OpenCLBuffer> buffer = nullptr;
    EventId eventId = nextEventId;

    if (kernelArgument.getMemoryLocation() == ArgumentMemoryLocation::HostZeroCopy)
    {
        buffer = std::make_unique<OpenCLBuffer>(context->getContext(), kernelArgument, true);
        bufferEvents.insert(std::make_pair(eventId, std::make_unique<OpenCLEvent>(eventId, false)));
    }
    else
    {
        buffer = std::make_unique<OpenCLBuffer>(context->getContext(), kernelArgument, false);
        auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);
        buffer->uploadData(commandQueues.at(queue)->getQueue(), kernelArgument.getData(), kernelArgument.getDataSizeInBytes(),
            profilingEvent->getEvent());

        profilingEvent->setReleaseFlag();
        bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    }

    buffers.insert(std::move(buffer)); // buffer data will be stolen
    nextEventId++;
    return eventId;
}

uint64_t OpenCLEngine::updateArgument(const ArgumentId id, const void* data, const size_t dataSizeInBytes)
{
    EventId eventId = updateArgumentAsync(id, data, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId OpenCLEngine::updateArgumentAsync(const ArgumentId id, const void* data, const size_t dataSizeInBytes, const QueueId queue)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    OpenCLBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);
    
    if (dataSizeInBytes == 0)
    {
        buffer->uploadData(commandQueues.at(queue)->getQueue(), data, buffer->getBufferSize(), profilingEvent->getEvent());
    }
    else
    {
        buffer->uploadData(commandQueues.at(queue)->getQueue(), data, dataSizeInBytes, profilingEvent->getEvent());
    }

    profilingEvent->setReleaseFlag();
    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    nextEventId++;
    return eventId;
}

uint64_t OpenCLEngine::downloadArgument(const ArgumentId id, void* destination, const size_t dataSizeInBytes) const
{
    EventId eventId = downloadArgumentAsync(id, destination, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId OpenCLEngine::downloadArgumentAsync(const ArgumentId id, void* destination, const size_t dataSizeInBytes, const QueueId queue) const
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    OpenCLBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);

    if (dataSizeInBytes == 0)
    {
        buffer->downloadData(commandQueues.at(queue)->getQueue(), destination, buffer->getBufferSize(), profilingEvent->getEvent());
    }
    else
    {
        buffer->downloadData(commandQueues.at(queue)->getQueue(), destination, dataSizeInBytes, profilingEvent->getEvent());
    }

    profilingEvent->setReleaseFlag();
    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    nextEventId++;
    return eventId;
}

KernelArgument OpenCLEngine::downloadArgumentObject(const ArgumentId id, uint64_t* downloadDuration) const
{
    OpenCLBuffer* buffer = findBuffer(id);

    if (buffer == nullptr)
    {
        throw std::runtime_error(std::string("Buffer with following id was not found: ") + std::to_string(id));
    }

    KernelArgument argument(buffer->getKernelArgumentId(), buffer->getBufferSize() / buffer->getElementSize(), buffer->getElementSize(),
        buffer->getDataType(), buffer->getMemoryLocation(), buffer->getAccessType(), ArgumentUploadType::Vector);

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);
    buffer->downloadData(commandQueues.at(getDefaultQueue())->getQueue(), argument.getData(), argument.getDataSizeInBytes(),
        profilingEvent->getEvent());

    profilingEvent->setReleaseFlag();
    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    nextEventId++;

    uint64_t duration = getArgumentOperationDuration(eventId);
    if (downloadDuration != nullptr)
    {
        *downloadDuration = duration;
    }
    
    return argument;
}

uint64_t OpenCLEngine::copyArgument(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes)
{
    EventId eventId = copyArgumentAsync(destination, source, dataSizeInBytes, getDefaultQueue());
    return getArgumentOperationDuration(eventId);
}

EventId OpenCLEngine::copyArgumentAsync(const ArgumentId destination, const ArgumentId source, const size_t dataSizeInBytes, const QueueId queue)
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    OpenCLBuffer* destinationBuffer = findBuffer(destination);
    OpenCLBuffer* sourceBuffer = findBuffer(source);

    if (destinationBuffer == nullptr || sourceBuffer == nullptr)
    {
        throw std::runtime_error(std::string("One of the buffers with following ids does not exist: ") + std::to_string(destination) + ", "
            + std::to_string(source));
    }

    if (sourceBuffer->getDataType() != destinationBuffer->getDataType())
    {
        throw std::runtime_error("Data type for buffers during copying operation must match");
    }

    EventId eventId = nextEventId;
    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, true);

    if (dataSizeInBytes == 0)
    {
        destinationBuffer->uploadData(commandQueues.at(queue)->getQueue(), sourceBuffer->getBuffer(), sourceBuffer->getBufferSize(),
            profilingEvent->getEvent());
    }
    else
    {
        destinationBuffer->uploadData(commandQueues.at(queue)->getQueue(), sourceBuffer->getBuffer(), dataSizeInBytes, profilingEvent->getEvent());
    }

    profilingEvent->setReleaseFlag();
    bufferEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    nextEventId++;
    return eventId;
}

uint64_t OpenCLEngine::getArgumentOperationDuration(const EventId id) const
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

    checkOpenCLError(clWaitForEvents(1, eventPointer->second->getEvent()), "clWaitForEvents");
    cl_ulong duration = eventPointer->second->getEventCommandDuration();
    bufferEvents.erase(id);

    return static_cast<uint64_t>(duration);
}

void OpenCLEngine::clearBuffer(const ArgumentId id)
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

void OpenCLEngine::clearBuffers()
{
    buffers.clear();
}

void OpenCLEngine::clearBuffers(const ArgumentAccessType accessType)
{
    auto iterator = buffers.cbegin();

    while (iterator != buffers.cend())
    {
        if (iterator->get()->getOpenclMemoryFlag() == getOpenCLMemoryType(accessType))
        {
            iterator = buffers.erase(iterator);
        }
        else
        {
            ++iterator;
        }
    }
}

void OpenCLEngine::printComputeAPIInfo(std::ostream& outputTarget) const
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

std::vector<PlatformInfo> OpenCLEngine::getPlatformInfo() const
{
    std::vector<PlatformInfo> result;
    auto platforms = getOpenCLPlatforms();

    for (size_t i = 0; i < platforms.size(); i++)
    {
        result.push_back(getOpenCLPlatformInfo(static_cast<PlatformIndex>(i)));
    }

    return result;
}

std::vector<DeviceInfo> OpenCLEngine::getDeviceInfo(const PlatformIndex platform) const
{
    std::vector<DeviceInfo> result;
    auto platforms = getOpenCLPlatforms();
    auto devices = getOpenCLDevices(platforms.at(platform));

    for (size_t i = 0; i < devices.size(); i++)
    {
        result.push_back(getOpenCLDeviceInfo(platform, static_cast<DeviceIndex>(i)));
    }

    return result;
}

DeviceInfo OpenCLEngine::getCurrentDeviceInfo() const
{
    return getOpenCLDeviceInfo(platformIndex, deviceIndex);
}

std::unique_ptr<OpenCLProgram> OpenCLEngine::createAndBuildProgram(const std::string& source) const
{
    auto program = std::make_unique<OpenCLProgram>(source, context->getContext(), context->getDevices());
    program->build(compilerOptions);
    return program;
}

void OpenCLEngine::setKernelArgument(OpenCLKernel& kernel, KernelArgument& argument)
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

void OpenCLEngine::setKernelArgument(OpenCLKernel& kernel, KernelArgument& argument, const std::vector<LocalMemoryModifier>& modifiers)
{
    size_t numberOfElements = argument.getNumberOfElements();

    for (const auto& modifier : modifiers)
    {
        if (modifier.getArgument() == argument.getId())
        {
            numberOfElements = modifier.getModifiedValue(numberOfElements);
        }
    }

    kernel.setKernelArgumentLocal(argument.getElementSizeInBytes() * numberOfElements);
}

EventId OpenCLEngine::enqueueKernel(OpenCLKernel& kernel, const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize,
    const QueueId queue, const uint64_t kernelLaunchOverhead) const
{
    if (queue >= commandQueues.size())
    {
        throw std::runtime_error(std::string("Invalid queue index: ") + std::to_string(queue));
    }

    std::vector<size_t> correctedGlobalSize = globalSize;
    if (globalSizeType == GlobalSizeType::CUDA)
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
    auto profilingEvent = std::make_unique<OpenCLEvent>(eventId, kernel.getKernelName(), kernelLaunchOverhead);
    nextEventId++;

    cl_int result = clEnqueueNDRangeKernel(commandQueues.at(queue)->getQueue(), kernel.getKernel(),
        static_cast<cl_uint>(correctedGlobalSize.size()), nullptr, correctedGlobalSize.data(), localSize.data(), 0, nullptr, profilingEvent->getEvent());
    checkOpenCLError(result, "clEnqueueNDRangeKernel");

    profilingEvent->setReleaseFlag();
    kernelEvents.insert(std::make_pair(eventId, std::move(profilingEvent)));
    return eventId;
}

PlatformInfo OpenCLEngine::getOpenCLPlatformInfo(const PlatformIndex platform)
{
    auto platforms = getOpenCLPlatforms();
    PlatformInfo result(platform, platforms.at(platform).getName());

    cl_platform_id id = platforms.at(platform).getId();
    result.setExtensions(getPlatformInfoString(id, CL_PLATFORM_EXTENSIONS));
    result.setVendor(getPlatformInfoString(id, CL_PLATFORM_VENDOR));
    result.setVersion(getPlatformInfoString(id, CL_PLATFORM_VERSION));

    return result;
}

DeviceInfo OpenCLEngine::getOpenCLDeviceInfo(const PlatformIndex platform, const DeviceIndex device)
{
    auto platforms = getOpenCLPlatforms();
    auto devices = getOpenCLDevices(platforms.at(platform));
    DeviceInfo result(device, devices.at(device).getName());

    cl_device_id id = devices.at(device).getId();
    result.setExtensions(getDeviceInfoString(id, CL_DEVICE_EXTENSIONS));
    result.setVendor(getDeviceInfoString(id, CL_DEVICE_VENDOR));
        
    uint64_t globalMemorySize;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(uint64_t), &globalMemorySize, nullptr));
    result.setGlobalMemorySize(globalMemorySize);

    uint64_t localMemorySize;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(uint64_t), &localMemorySize, nullptr));
    result.setLocalMemorySize(localMemorySize);

    uint64_t maxConstantBufferSize;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(uint64_t), &maxConstantBufferSize, nullptr));
    result.setMaxConstantBufferSize(maxConstantBufferSize);

    uint32_t maxComputeUnits;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint32_t), &maxComputeUnits, nullptr));
    result.setMaxComputeUnits(maxComputeUnits);

    size_t maxWorkGroupSize;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr));
    result.setMaxWorkGroupSize(maxWorkGroupSize);

    cl_device_type deviceType;
    checkOpenCLError(clGetDeviceInfo(id, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
    result.setDeviceType(getDeviceType(deviceType));

    return result;
}

std::vector<OpenCLPlatform> OpenCLEngine::getOpenCLPlatforms()
{
    cl_uint platformCount;
    checkOpenCLError(clGetPlatformIDs(0, nullptr, &platformCount));

    std::vector<cl_platform_id> platformIds(platformCount);
    checkOpenCLError(clGetPlatformIDs(platformCount, platformIds.data(), nullptr));

    std::vector<OpenCLPlatform> platforms;
    for (const auto platformId : platformIds)
    {
        std::string name = getPlatformInfoString(platformId, CL_PLATFORM_NAME);
        platforms.push_back(OpenCLPlatform(platformId, name));
    }

    return platforms;
}

std::vector<OpenCLDevice> OpenCLEngine::getOpenCLDevices(const OpenCLPlatform& platform)
{
    cl_uint deviceCount;
    checkOpenCLError(clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount));

    std::vector<cl_device_id> deviceIds(deviceCount);
    checkOpenCLError(clGetDeviceIDs(platform.getId(), CL_DEVICE_TYPE_ALL, deviceCount, deviceIds.data(), nullptr));

    std::vector<OpenCLDevice> devices;
    for (const auto deviceId : deviceIds)
    {
        std::string name = getDeviceInfoString(deviceId, CL_DEVICE_NAME);
        devices.push_back(OpenCLDevice(deviceId, name));
    }

    return devices;
}

DeviceType OpenCLEngine::getDeviceType(const cl_device_type deviceType)
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

OpenCLBuffer* OpenCLEngine::findBuffer(const ArgumentId id) const
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

void OpenCLEngine::setKernelArgumentVector(OpenCLKernel& kernel, const OpenCLBuffer& buffer) const
{
    cl_mem clBuffer = buffer.getBuffer();
    kernel.setKernelArgumentVector((void*)&clBuffer);
}

bool OpenCLEngine::loadBufferFromCache(const ArgumentId id, OpenCLKernel& kernel) const
{
    OpenCLBuffer* buffer = findBuffer(id);

    if (buffer != nullptr)
    {
        setKernelArgumentVector(kernel, *buffer);
        return true;
    }

    return false;
}

void OpenCLEngine::checkLocalMemoryModifiers(const std::vector<KernelArgument*>& argumentPointers,
    const std::vector<LocalMemoryModifier>& modifiers) const
{
    for (const auto& modifier : modifiers)
    {
        bool modifierArgumentFound = false;

        for (const auto argument : argumentPointers)
        {
            if (modifier.getArgument() == argument->getId() && argument->getUploadType() == ArgumentUploadType::Local)
            {
                modifierArgumentFound = true;
            }
        }

        if (!modifierArgumentFound)
        {
            throw std::runtime_error(std::string("No matching local memory argument found for modifier, argument id in modifier: ")
                + std::to_string(modifier.getArgument()));
        }
    }
}

#else

OpenCLEngine::OpenCLEngine(const PlatformIndex, const DeviceIndex, const uint32_t)
{
    throw std::runtime_error("Support for OpenCL API is not included in this version of KTT framework");
}

KernelResult OpenCLEngine::runKernel(const KernelRuntimeData&, const std::vector<KernelArgument*>&, const std::vector<OutputDescriptor>&)
{
    throw std::runtime_error("");
}

EventId OpenCLEngine::runKernelAsync(const KernelRuntimeData&, const std::vector<KernelArgument*>&, const QueueId)
{
    throw std::runtime_error("");
}

KernelResult OpenCLEngine::getKernelResult(const EventId, const std::vector<OutputDescriptor>&) const
{
    throw std::runtime_error("");
}

void OpenCLEngine::setCompilerOptions(const std::string&)
{
    throw std::runtime_error("");
}

void OpenCLEngine::setGlobalSizeType(const GlobalSizeType)
{
    throw std::runtime_error("");
}

void OpenCLEngine::setAutomaticGlobalSizeCorrection(const bool)
{
    throw std::runtime_error("");
}

void OpenCLEngine::setProgramCache(const bool)
{
    throw std::runtime_error("");
}

void OpenCLEngine::clearProgramCache()
{
    throw std::runtime_error("");
}

QueueId OpenCLEngine::getDefaultQueue() const
{
    throw std::runtime_error("");
}

std::vector<QueueId> OpenCLEngine::getAllQueues() const
{
    throw std::runtime_error("");
}

void OpenCLEngine::synchronizeQueue(const QueueId)
{
    throw std::runtime_error("");
}

void OpenCLEngine::synchronizeDevice()
{
    throw std::runtime_error("");
}

void OpenCLEngine::clearEvents()
{
    throw std::runtime_error("");
}

uint64_t OpenCLEngine::uploadArgument(KernelArgument&)
{
    throw std::runtime_error("");
}

EventId OpenCLEngine::uploadArgumentAsync(KernelArgument&, const QueueId)
{
    throw std::runtime_error("");
}

uint64_t OpenCLEngine::updateArgument(const ArgumentId, const void*, const size_t)
{
    throw std::runtime_error("");
}

EventId OpenCLEngine::updateArgumentAsync(const ArgumentId, const void*, const size_t, const QueueId)
{
    throw std::runtime_error("");
}

uint64_t OpenCLEngine::downloadArgument(const ArgumentId, void*, const size_t) const
{
    throw std::runtime_error("");
}

EventId OpenCLEngine::downloadArgumentAsync(const ArgumentId, void*, const size_t, const QueueId) const
{
    throw std::runtime_error("");
}

KernelArgument OpenCLEngine::downloadArgumentObject(const ArgumentId, uint64_t*) const
{
    throw std::runtime_error("");
}

uint64_t OpenCLEngine::copyArgument(const ArgumentId, const ArgumentId, const size_t)
{
    throw std::runtime_error("");
}

EventId OpenCLEngine::copyArgumentAsync(const ArgumentId, const ArgumentId, const size_t, const QueueId)
{
    throw std::runtime_error("");
}

uint64_t OpenCLEngine::getArgumentOperationDuration(const EventId) const
{
    throw std::runtime_error("");
}

void OpenCLEngine::clearBuffer(const ArgumentId)
{
    throw std::runtime_error("");
}

void OpenCLEngine::clearBuffers()
{
    throw std::runtime_error("");
}

void OpenCLEngine::clearBuffers(const ArgumentAccessType)
{
    throw std::runtime_error("");
}

void OpenCLEngine::printComputeAPIInfo(std::ostream&) const
{
    throw std::runtime_error("");
}

std::vector<PlatformInfo> OpenCLEngine::getPlatformInfo() const
{
    throw std::runtime_error("");
}

std::vector<DeviceInfo> OpenCLEngine::getDeviceInfo(const PlatformIndex) const
{
    throw std::runtime_error("");
}

DeviceInfo OpenCLEngine::getCurrentDeviceInfo() const
{
    throw std::runtime_error("");
}

#endif // PLATFORM_OPENCL

} // namespace ktt
