#include <stdexcept>
#include <utility>
#include <tuning_runner/manipulator_interface_implementation.h>
#include <utility/ktt_utility.h>
#include <utility/timer.h>

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(ComputeEngine* computeEngine) :
    computeEngine(computeEngine),
    currentConfiguration(KernelConfiguration()),
    currentResult("", currentConfiguration),
    kernelProfilingFlag(false)
{}

void ManipulatorInterfaceImplementation::runKernel(const KernelId id)
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }
    runKernel(id, dataPointer->second.getGlobalSizeDimensionVector(), dataPointer->second.getLocalSizeDimensionVector());
}

void ManipulatorInterfaceImplementation::runKernelAsync(const KernelId id, const QueueId queue)
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }
    runKernelAsync(id, dataPointer->second.getGlobalSizeDimensionVector(), dataPointer->second.getLocalSizeDimensionVector(), queue);
}

void ManipulatorInterfaceImplementation::runKernel(const KernelId id, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    KernelRuntimeData kernelData = dataPointer->second;
    kernelData.setGlobalSize(globalSize);
    kernelData.setLocalSize(localSize);

    KernelResult result = computeEngine->runKernel(kernelData, getArgumentPointers(kernelData.getArgumentIds()), std::vector<OutputDescriptor>{});
    currentResult.increaseOverhead(result.getOverhead());
}

void ManipulatorInterfaceImplementation::runKernelAsync(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize,
    const QueueId queue)
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    KernelRuntimeData kernelData = dataPointer->second;
    kernelData.setGlobalSize(globalSize);
    kernelData.setLocalSize(localSize);

    EventId kernelEvent = computeEngine->runKernelAsync(kernelData, getArgumentPointers(kernelData.getArgumentIds()), queue);
    storeKernelEvent(queue, kernelEvent);
    currentResult.increaseOverhead(computeEngine->getKernelOverhead(kernelEvent));
}

void ManipulatorInterfaceImplementation::runKernelWithProfiling(const KernelId id)
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }
    runKernelWithProfiling(id, dataPointer->second.getGlobalSizeDimensionVector(), dataPointer->second.getLocalSizeDimensionVector());
}

void ManipulatorInterfaceImplementation::runKernelWithProfiling(const KernelId id, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    if (!kernelProfilingFlag)
    {
        throw std::runtime_error("Calling kernel profiling methods from tuning manipulator is not allowed when kernel profiling is disabled");
    }

    if (profiledKernels.find(id) == profiledKernels.end())
    {
        throw std::runtime_error(std::string("Kernel profiling for kernel with the following id is disabled: ") + std::to_string(id));
    }

    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    KernelRuntimeData kernelData = dataPointer->second;
    kernelData.setGlobalSize(globalSize);
    kernelData.setLocalSize(localSize);

    EventId kernelEvent = computeEngine->runKernelWithProfiling(kernelData, getArgumentPointers(kernelData.getArgumentIds()),
        getDefaultDeviceQueue());
    storeKernelProfilingEvent(id, kernelEvent);
    currentResult.increaseOverhead(computeEngine->getKernelOverhead(kernelEvent));
}

uint64_t ManipulatorInterfaceImplementation::getRemainingKernelProfilingRuns(const KernelId id) const
{
    if (!kernelProfilingFlag)
    {
        throw std::runtime_error("Calling kernel profiling methods from tuning manipulator is not allowed when kernel profiling is disabled");
    }

    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }
    return computeEngine->getRemainingKernelProfilingRuns(dataPointer->second.getName(), dataPointer->second.getSource());
}

QueueId ManipulatorInterfaceImplementation::getDefaultDeviceQueue() const
{
    return computeEngine->getDefaultQueue();
}

std::vector<QueueId> ManipulatorInterfaceImplementation::getAllDeviceQueues() const
{
    return computeEngine->getAllQueues();
}

void ManipulatorInterfaceImplementation::synchronizeQueue(const QueueId queue)
{
    computeEngine->synchronizeQueue(queue);

    auto eventPointer = enqueuedKernelEvents.find(queue);
    if (eventPointer != enqueuedKernelEvents.end())
    {
        processKernelEvents(eventPointer->second);
        enqueuedKernelEvents.erase(queue);
    }

    auto bufferEventPointer = enqueuedBufferEvents.find(queue);
    if (bufferEventPointer != enqueuedBufferEvents.end())
    {
        processBufferEvents(bufferEventPointer->second);
        enqueuedBufferEvents.erase(queue);
    }
}

void ManipulatorInterfaceImplementation::synchronizeDevice()
{
    computeEngine->synchronizeDevice();

    for (const auto& queueEventPair : enqueuedKernelEvents)
    {
        processKernelEvents(queueEventPair.second);
    }
    enqueuedKernelEvents.clear();

    for (const auto& queueEventPair : enqueuedBufferEvents)
    {
        processBufferEvents(queueEventPair.second);
    }
    enqueuedBufferEvents.clear();
}

DimensionVector ManipulatorInterfaceImplementation::getCurrentGlobalSize(const KernelId id) const
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }
    return dataPointer->second.getGlobalSizeDimensionVector();
}

DimensionVector ManipulatorInterfaceImplementation::getCurrentLocalSize(const KernelId id) const
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }
    return dataPointer->second.getLocalSizeDimensionVector();
}

std::vector<ParameterPair> ManipulatorInterfaceImplementation::getCurrentConfiguration() const
{
    return currentConfiguration.getParameterPairs();
}

void ManipulatorInterfaceImplementation::updateArgumentScalar(const ArgumentId id, const void* argumentData)
{
    updateArgumentSimple(id, argumentData, 1, ArgumentUploadType::Scalar);
}

void ManipulatorInterfaceImplementation::updateArgumentLocal(const ArgumentId id, const size_t numberOfElements)
{
    updateArgumentSimple(id, nullptr, numberOfElements, ArgumentUploadType::Local);
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const ArgumentId id, const void* argumentData)
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    computeEngine->updateArgument(id, argumentData, 0);
}

void ManipulatorInterfaceImplementation::updateArgumentVectorAsync(const ArgumentId id, const void* argumentData, const QueueId queue)
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    EventId bufferEvent = computeEngine->updateArgumentAsync(id, argumentData, 0, queue);
    storeBufferEvent(queue, bufferEvent, false);
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements)
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    computeEngine->updateArgument(id, argumentData, argumentPointer->second->getElementSizeInBytes() * numberOfElements);
}

void ManipulatorInterfaceImplementation::updateArgumentVectorAsync(const ArgumentId id, const void* argumentData, const size_t numberOfElements,
    const QueueId queue)
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    EventId bufferEvent = computeEngine->updateArgumentAsync(id, argumentData, argumentPointer->second->getElementSizeInBytes() * numberOfElements,
        queue);
    storeBufferEvent(queue, bufferEvent, false);
}

void ManipulatorInterfaceImplementation::getArgumentVector(const ArgumentId id, void* destination) const
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    computeEngine->downloadArgument(id, destination, 0);
}

void ManipulatorInterfaceImplementation::getArgumentVectorAsync(const ArgumentId id, void* destination, const QueueId queue) const
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    EventId bufferEvent = computeEngine->downloadArgumentAsync(id, destination, 0, queue);
    storeBufferEvent(queue, bufferEvent, false);
}

void ManipulatorInterfaceImplementation::getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    computeEngine->downloadArgument(id, destination, argumentPointer->second->getElementSizeInBytes() * numberOfElements);
}

void ManipulatorInterfaceImplementation::getArgumentVectorAsync(const ArgumentId id, void* destination, const size_t numberOfElements,
    const QueueId queue) const
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    EventId bufferEvent = computeEngine->downloadArgumentAsync(id, destination, argumentPointer->second->getElementSizeInBytes() * numberOfElements,
        queue);
    storeBufferEvent(queue, bufferEvent, false);
}

void ManipulatorInterfaceImplementation::copyArgumentVector(const ArgumentId destination, const ArgumentId source, const size_t numberOfElements)
{
    auto argumentPointer = vectorArguments.find(destination);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(destination));
    }

    auto argumentPointerSource = vectorArguments.find(source);
    if (argumentPointerSource == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(source));
    }

    computeEngine->copyArgument(destination, source, argumentPointer->second->getElementSizeInBytes() * numberOfElements);
}

void ManipulatorInterfaceImplementation::copyArgumentVectorAsync(const ArgumentId destination, const ArgumentId source,
    const size_t numberOfElements, const QueueId queue)
{
    auto argumentPointer = vectorArguments.find(destination);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(destination));
    }

    auto argumentPointerSource = vectorArguments.find(source);
    if (argumentPointerSource == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(source));
    }

    computeEngine->copyArgumentAsync(destination, source, argumentPointer->second->getElementSizeInBytes() * numberOfElements, queue);
}

void ManipulatorInterfaceImplementation::resizeArgumentVector(const ArgumentId id, const size_t newNumberOfElements, const bool preserveOldData)
{
    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    computeEngine->resizeArgument(id, argumentPointer->second->getElementSizeInBytes() * newNumberOfElements, preserveOldData);
}

void ManipulatorInterfaceImplementation::changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    if (!containsUnique(argumentIds))
    {
        throw std::runtime_error("Kernel argument ids assigned to single kernel must be unique");
    }

    dataPointer->second.setArgumentIndices(argumentIds);
}

void ManipulatorInterfaceImplementation::swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond)
{
    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    std::vector<ArgumentId> argumentIds = dataPointer->second.getArgumentIds();
    
    if (!elementExists(argumentIdFirst, argumentIds) || !elementExists(argumentIdSecond, argumentIds))
    {
        throw std::runtime_error(std::string("One of the following argument ids is not associated with this kernel: ")
            + std::to_string(argumentIdFirst) + ", " + std::to_string(argumentIdSecond) + ", kernel id: " + std::to_string(id));
    }

    ArgumentId firstId;
    ArgumentId secondId;
    for (size_t i = 0; i < argumentIds.size(); i++)
    {
        if (argumentIds.at(i) == argumentIdFirst)
        {
            firstId = i;
        }
        if (argumentIds.at(i) == argumentIdSecond)
        {
            secondId = i;
        }
    }
    std::swap(argumentIds.at(firstId), argumentIds.at(secondId));

    dataPointer->second.setArgumentIndices(argumentIds);
}

void ManipulatorInterfaceImplementation::createArgumentBuffer(const ArgumentId id)
{
    Timer timer;
    timer.start();

    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }
    computeEngine->uploadArgument(*argumentPointer->second);

    timer.stop();
    currentResult.increaseOverhead(timer.getElapsedTime());
}

void ManipulatorInterfaceImplementation::createArgumentBufferAsync(const ArgumentId id, const QueueId queue)
{
    Timer timer;
    timer.start();

    auto argumentPointer = vectorArguments.find(id);
    if (argumentPointer == vectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    EventId bufferEvent = computeEngine->uploadArgumentAsync(*argumentPointer->second, queue);
    storeBufferEvent(queue, bufferEvent, true);

    timer.stop();
    currentResult.increaseOverhead(timer.getElapsedTime());
}

void ManipulatorInterfaceImplementation::destroyArgumentBuffer(const ArgumentId id)
{
    Timer timer;
    timer.start();

    computeEngine->clearBuffer(id);

    timer.stop();
    currentResult.increaseOverhead(timer.getElapsedTime());
}

void ManipulatorInterfaceImplementation::setKernelProfiling(const bool flag)
{
    kernelProfilingFlag = flag;
}

void ManipulatorInterfaceImplementation::addKernel(const KernelId id, const KernelRuntimeData& data)
{
    kernelData.insert(std::make_pair(id, data));
}

void ManipulatorInterfaceImplementation::setConfiguration(const KernelConfiguration& configuration)
{
    currentConfiguration = configuration;
    currentResult.setComputationDuration(0);
    currentResult.setConfiguration(currentConfiguration);
    currentResult.setValid(true);
}

void ManipulatorInterfaceImplementation::setKernelArguments(const std::vector<KernelArgument*>& arguments)
{
    for (const auto& kernelArgument : arguments)
    {
        if (kernelArgument->getUploadType() == ArgumentUploadType::Vector)
        {
            vectorArguments.insert(std::make_pair(kernelArgument->getId(), kernelArgument));
        }
        else
        {
            nonVectorArguments.insert(std::make_pair(kernelArgument->getId(), *kernelArgument));
        }
    }
}

void ManipulatorInterfaceImplementation::uploadBuffers()
{
    for (auto& argument : vectorArguments)
    {
        if (!argument.second->isPersistent())
        {
            computeEngine->uploadArgument(*argument.second);
        }
    }
}

void ManipulatorInterfaceImplementation::downloadBuffers(const std::vector<OutputDescriptor>& output) const
{
    for (const auto& descriptor : output)
    {
        computeEngine->downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
    }
}

void ManipulatorInterfaceImplementation::synchronizeDeviceInternal()
{
    if (!enqueuedKernelEvents.empty() || !enqueuedBufferEvents.empty())
    {
        synchronizeDevice();
    }
}

void ManipulatorInterfaceImplementation::clearData()
{
    currentResult = KernelResult();
    currentConfiguration = KernelConfiguration();
    kernelData.clear();
    vectorArguments.clear();
    nonVectorArguments.clear();
    kernelProfilingEvents.clear();
    profiledKernels.clear();
}

void ManipulatorInterfaceImplementation::resetOverhead()
{
    currentResult.setOverhead(0);
}

void ManipulatorInterfaceImplementation::setProfiledKernels(const std::set<KernelId>& profiledKernels)
{
    this->profiledKernels = profiledKernels;
}

KernelResult ManipulatorInterfaceImplementation::getCurrentResult() const
{
    KernelResult result = currentResult;

    if (!kernelProfilingEvents.empty())
    {
        for (const auto& kernelEvents : kernelProfilingEvents)
        {
            KernelResult profilingResult = computeEngine->getKernelResultWithProfiling(kernelEvents.second[0], std::vector<OutputDescriptor>{});
            if (kernelData.size() == 1)
            {
                result.setProfilingData(profilingResult.getProfilingData());
            }
            else
            {
                result.setCompositionKernelProfilingData(kernelEvents.first, profilingResult.getProfilingData());
            }
        }

        kernelProfilingEvents.clear();
    }

    return result;
}

KernelResult ManipulatorInterfaceImplementation::getCurrentResult(const uint64_t remainingProfilingRuns) const
{
    if (remainingProfilingRuns == 0)
    {
        return getCurrentResult();
    }

    KernelResult result = currentResult;
    result.setProfilingData(KernelProfilingData(remainingProfilingRuns));
    return result;
}

std::vector<KernelArgument*> ManipulatorInterfaceImplementation::getArgumentPointers(const std::vector<ArgumentId>& argumentIds)
{
    std::vector<KernelArgument*> result;

    for (const auto id : argumentIds)
    {
        bool argumentAdded = false;

        for (const auto argument : vectorArguments)
        {
            if (id == argument.second->getId())
            {
                result.push_back(argument.second);
                argumentAdded = true;
                break;
            }
        }

        if (argumentAdded)
        {
            continue;
        }

        for (auto& argument : nonVectorArguments)
        {
            if (id == argument.second.getId())
            {
                result.push_back(&argument.second);
                break;
            }
        }
    }

    return result;
}

void ManipulatorInterfaceImplementation::updateArgumentSimple(const ArgumentId id, const void* argumentData, const size_t numberOfElements,
    const ArgumentUploadType uploadType)
{
    auto argumentPointer = nonVectorArguments.find(id);
    if (argumentPointer == nonVectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    if (argumentPointer->second.getUploadType() != uploadType)
    {
        throw std::runtime_error("Cannot convert between vector and non-vector arguments");
    }

    auto updatedArgument = KernelArgument(id, argumentData, numberOfElements, argumentPointer->second.getElementSizeInBytes(),
        argumentPointer->second.getDataType(), argumentPointer->second.getMemoryLocation(), argumentPointer->second.getAccessType(), uploadType);

    nonVectorArguments.erase(id);
    nonVectorArguments.insert(std::make_pair(id, updatedArgument));
}

void ManipulatorInterfaceImplementation::storeKernelEvent(const QueueId queue, const EventId event) const
{
    auto eventPointer = enqueuedKernelEvents.find(queue);
    if (eventPointer == enqueuedKernelEvents.end())
    {
        enqueuedKernelEvents.insert(std::make_pair(queue, std::set<EventId>{event}));
    }
    else
    {
        eventPointer->second.insert(event);
    }
}

void ManipulatorInterfaceImplementation::storeKernelProfilingEvent(const KernelId kernel, const EventId event) const
{
    auto eventPointer = kernelProfilingEvents.find(kernel);
    if (eventPointer == kernelProfilingEvents.end())
    {
        kernelProfilingEvents.insert(std::make_pair(kernel, std::vector<EventId>{event}));
    }
    else
    {
        eventPointer->second.push_back(event);
    }
}

void ManipulatorInterfaceImplementation::storeBufferEvent(const QueueId queue, const EventId event, const bool increaseOverhead) const
{
    auto eventPointer = enqueuedBufferEvents.find(queue);
    if (eventPointer == enqueuedBufferEvents.end())
    {
        enqueuedBufferEvents.insert(std::make_pair(queue, std::set<std::pair<EventId, bool>>{std::make_pair(event, increaseOverhead)}));
    }
    else
    {
        eventPointer->second.insert(std::make_pair(event, increaseOverhead));
    }
}

void ManipulatorInterfaceImplementation::processKernelEvents(const std::set<EventId>& events)
{
    for (const auto& currentEvent : events)
    {
        computeEngine->getKernelResult(currentEvent, std::vector<OutputDescriptor>{});
    }
}

void ManipulatorInterfaceImplementation::processBufferEvents(const std::set<std::pair<EventId, bool>>& events)
{
    for (const auto& currentEvent : events)
    {
        uint64_t result = computeEngine->getArgumentOperationDuration(currentEvent.first);
        if (currentEvent.second)
        {
            currentResult.increaseOverhead(result);
        }
    }
}

} // namespace ktt
