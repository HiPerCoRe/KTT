#include "tuning_manipulator.h"
#include "tuning_runner/manipulator_interface.h"

namespace ktt
{

TuningManipulator::~TuningManipulator() = default;

std::vector<std::pair<size_t, ThreadSizeUsage>> TuningManipulator::getUtilizedKernelIds() const
{
    return std::vector<std::pair<size_t, ThreadSizeUsage>>{};
}

void TuningManipulator::runKernel(const size_t kernelId)
{
    manipulatorInterface->runKernel(kernelId);
}

void TuningManipulator::runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize)
{
    manipulatorInterface->runKernel(kernelId, globalSize, localSize);
}

DimensionVector TuningManipulator::getCurrentGlobalSize(const size_t kernelId) const
{
    return manipulatorInterface->getCurrentGlobalSize(kernelId);
}

DimensionVector TuningManipulator::getCurrentLocalSize(const size_t kernelId) const
{
    return manipulatorInterface->getCurrentLocalSize(kernelId);
}

std::vector<ParameterValue> TuningManipulator::getCurrentConfiguration() const
{
    return manipulatorInterface->getCurrentConfiguration();
}

void TuningManipulator::updateArgumentScalar(const size_t argumentId, const void* argumentData)
{
    manipulatorInterface->updateArgumentScalar(argumentId, argumentData);
}

void TuningManipulator::updateArgumentVector(const size_t argumentId, const void* argumentData, const ArgumentLocation& argumentLocation)
{
    manipulatorInterface->updateArgumentVector(argumentId, argumentData, argumentLocation);
}

void TuningManipulator::updateArgumentVector(const size_t argumentId, const void* argumentData, const ArgumentLocation& argumentLocation,
    const size_t numberOfElements)
{
    manipulatorInterface->updateArgumentVector(argumentId, argumentData, argumentLocation, numberOfElements);
}

void TuningManipulator::synchronizeArgumentVector(const size_t argumentId, const bool downloadToHost)
{
    manipulatorInterface->synchronizeArgumentVector(argumentId, downloadToHost);
}

void TuningManipulator::setKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)
{
    manipulatorInterface->setKernelArguments(kernelId, argumentIds);
}

void TuningManipulator::swapKernelArguments(const size_t kernelId, const size_t argumentIdFirst, const size_t argumentIdSecond)
{
    manipulatorInterface->swapKernelArguments(kernelId, argumentIdFirst, argumentIdSecond);
}

std::vector<size_t> TuningManipulator::convertFromDimensionVector(const DimensionVector& vector)
{
    std::vector<size_t> result;

    result.push_back(std::get<0>(vector));
    result.push_back(std::get<1>(vector));
    result.push_back(std::get<2>(vector));

    return result;
}

DimensionVector TuningManipulator::convertToDimensionVector(const std::vector<size_t>& vector)
{
    if (vector.size() > 2)
    {
        return DimensionVector(vector.at(0), vector.at(1), vector.at(2));
    }
    if (vector.size() > 1)
    {
        return DimensionVector(vector.at(0), vector.at(1), 1);
    }
    if (vector.size() > 1)
    {
        return DimensionVector(vector.at(0), 1, 1);
    }
    return DimensionVector(1, 1, 1);
}

size_t TuningManipulator::getParameterValue(const std::string& parameterName, const std::vector<ParameterValue>& parameterValues)
{
    for (const auto& parameterValue : parameterValues)
    {
        if (std::get<0>(parameterValue) == parameterName)
        {
            return std::get<1>(parameterValue);
        }
    }
    throw std::runtime_error(std::string("No parameter with following name found: ") + parameterName);
}

} // namespace ktt
