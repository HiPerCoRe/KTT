#include <api/tuning_manipulator.h>
#include <tuning_runner/manipulator_interface.h>

namespace ktt
{

TuningManipulator::~TuningManipulator() = default;

bool TuningManipulator::enableArgumentPreload() const
{
    return true;
}

void TuningManipulator::runKernel(const KernelId id)
{
    manipulatorInterface->runKernel(id);
}

void TuningManipulator::runKernelAsync(const KernelId id, const QueueId queue)
{
    manipulatorInterface->runKernelAsync(id, queue);
}

void TuningManipulator::runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize)
{
    manipulatorInterface->runKernel(id, globalSize, localSize);
}

void TuningManipulator::runKernelAsync(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize, const QueueId queue)
{
    manipulatorInterface->runKernelAsync(id, globalSize, localSize, queue);
}

void TuningManipulator::runKernelWithProfiling(const KernelId id)
{
    manipulatorInterface->runKernelWithProfiling(id);
}

void TuningManipulator::runKernelWithProfiling(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize)
{
    manipulatorInterface->runKernelWithProfiling(id, globalSize, localSize);
}

uint64_t TuningManipulator::getRemainingKernelProfilingRuns(const KernelId id) const
{
    return manipulatorInterface->getRemainingKernelProfilingRuns(id);
}

QueueId TuningManipulator::getDefaultDeviceQueue() const
{
    return manipulatorInterface->getDefaultDeviceQueue();
}

std::vector<QueueId> TuningManipulator::getAllDeviceQueues() const
{
    return manipulatorInterface->getAllDeviceQueues();
}

void TuningManipulator::synchronizeQueue(const QueueId queue)
{
    manipulatorInterface->synchronizeQueue(queue);
}

void TuningManipulator::synchronizeDevice()
{
    manipulatorInterface->synchronizeDevice();
}

DimensionVector TuningManipulator::getCurrentGlobalSize(const KernelId id) const
{
    return manipulatorInterface->getCurrentGlobalSize(id);
}

DimensionVector TuningManipulator::getCurrentLocalSize(const KernelId id) const
{
    return manipulatorInterface->getCurrentLocalSize(id);
}

std::vector<ParameterPair> TuningManipulator::getCurrentConfiguration() const
{
    return manipulatorInterface->getCurrentConfiguration();
}

void TuningManipulator::updateArgumentScalar(const ArgumentId id, const void* argumentData)
{
    manipulatorInterface->updateArgumentScalar(id, argumentData);
}

void TuningManipulator::updateArgumentLocal(const ArgumentId id, const size_t numberOfElements)
{
    manipulatorInterface->updateArgumentLocal(id, numberOfElements);
}

void TuningManipulator::updateArgumentVector(const ArgumentId id, const void* argumentData)
{
    manipulatorInterface->updateArgumentVector(id, argumentData);
}

void TuningManipulator::updateArgumentVectorAsync(const ArgumentId id, const void* argumentData, QueueId queue)
{
    manipulatorInterface->updateArgumentVectorAsync(id, argumentData, queue);
}

void TuningManipulator::updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements)
{
    manipulatorInterface->updateArgumentVector(id, argumentData, numberOfElements);
}

void TuningManipulator::updateArgumentVectorAsync(const ArgumentId id, const void* argumentData, const size_t numberOfElements, QueueId queue)
{
    manipulatorInterface->updateArgumentVectorAsync(id, argumentData, numberOfElements, queue);
}

void TuningManipulator::getArgumentVector(const ArgumentId id, void* destination) const
{
    manipulatorInterface->getArgumentVector(id, destination);
}

void TuningManipulator::getArgumentVectorAsync(const ArgumentId id, void* destination, QueueId queue) const
{
    manipulatorInterface->getArgumentVectorAsync(id, destination, queue);
}

void TuningManipulator::getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const
{
    manipulatorInterface->getArgumentVector(id, destination, numberOfElements);
}

void TuningManipulator::getArgumentVectorAsync(const ArgumentId id, void* destination, const size_t numberOfElements, QueueId queue) const
{
    manipulatorInterface->getArgumentVectorAsync(id, destination, numberOfElements, queue);
}

void TuningManipulator::copyArgumentVector(const ArgumentId destination, const ArgumentId source, const size_t numberOfElements)
{
    manipulatorInterface->copyArgumentVector(destination, source, numberOfElements);
}

void TuningManipulator::copyArgumentVectorAsync(const ArgumentId destination, const ArgumentId source, const size_t numberOfElements,
    const QueueId queue)
{
    manipulatorInterface->copyArgumentVectorAsync(destination, source, numberOfElements, queue);
}

void TuningManipulator::resizeArgumentVector(const ArgumentId id, const size_t newNumberOfElements, const bool preserveOldData)
{
    manipulatorInterface->resizeArgumentVector(id, newNumberOfElements, preserveOldData);
}

void TuningManipulator::changeKernelArguments(const KernelId id, const std::vector<ArgumentId>& argumentIds)
{
    manipulatorInterface->changeKernelArguments(id, argumentIds);
}

void TuningManipulator::swapKernelArguments(const KernelId id, const ArgumentId argumentIdFirst, const ArgumentId argumentIdSecond)
{
    manipulatorInterface->swapKernelArguments(id, argumentIdFirst, argumentIdSecond);
}

void TuningManipulator::createArgumentBuffer(const ArgumentId id)
{
    manipulatorInterface->createArgumentBuffer(id);
}

void TuningManipulator::createArgumentBufferAsync(const ArgumentId id, QueueId queue)
{
    manipulatorInterface->createArgumentBufferAsync(id, queue);
}

void TuningManipulator::destroyArgumentBuffer(const ArgumentId id)
{
    manipulatorInterface->destroyArgumentBuffer(id);
}

size_t TuningManipulator::getParameterValue(const std::string& parameterName, const std::vector<ParameterPair>& parameterPairs)
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (parameterPair.getName() == parameterName)
        {
            return parameterPair.getValue();
        }
    }
    throw std::runtime_error(std::string("No parameter with following name found: ") + parameterName);
}

double TuningManipulator::getParameterValueDouble(const std::string& parameterName, const std::vector<ParameterPair>& parameterPairs)
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (parameterPair.getName() == parameterName)
        {
            return parameterPair.getValueDouble();
        }
    }
    throw std::runtime_error(std::string("No parameter with following name found: ") + parameterName);
}

} // namespace ktt
