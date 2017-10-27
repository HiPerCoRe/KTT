#include "tuning_manipulator.h"
#include "tuning_runner/manipulator_interface.h"

namespace ktt
{

TuningManipulator::~TuningManipulator() = default;

void TuningManipulator::runKernel(const KernelId id)
{
    manipulatorInterface->runKernel(id);
}

TunerFlag TuningManipulator::enableArgumentPreload() const
{
    return true;
}

void TuningManipulator::runKernel(const KernelId id, const DimensionVector& globalSize, const DimensionVector& localSize)
{
    manipulatorInterface->runKernel(id, globalSize, localSize);
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

void TuningManipulator::updateArgumentVector(const ArgumentId id, const void* argumentData, const size_t numberOfElements)
{
    manipulatorInterface->updateArgumentVector(id, argumentData, numberOfElements);
}

void TuningManipulator::getArgumentVector(const ArgumentId id, void* destination) const
{
    manipulatorInterface->getArgumentVector(id, destination);
}

void TuningManipulator::getArgumentVector(const ArgumentId id, void* destination, const size_t numberOfElements) const
{
    manipulatorInterface->getArgumentVector(id, destination, numberOfElements);
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

void TuningManipulator::destroyArgumentBuffer(const ArgumentId id)
{
    manipulatorInterface->destroyArgumentBuffer(id);
}

size_t TuningManipulator::getParameterValue(const std::string& parameterName, const std::vector<ParameterPair>& parameterPairs)
{
    for (const auto& parameterPair : parameterPairs)
    {
        if (std::get<0>(parameterPair) == parameterName)
        {
            return std::get<1>(parameterPair);
        }
    }
    throw std::runtime_error(std::string("No parameter with following name found: ") + parameterName);
}

} // namespace ktt
