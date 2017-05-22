#include "tuning_manipulator.h"

namespace ktt
{

TuningManipulator::~TuningManipulator() = default;

std::vector<std::pair<size_t, ThreadSizeUsage>> TuningManipulator::getUtilizedKernelIds() const
{
    return std::vector<std::pair<size_t, ThreadSizeUsage>>{};
}

std::vector<ResultArgument> TuningManipulator::runKernel(const size_t kernelId)
{
    return manipulatorInterface->runKernel(kernelId);
}

std::vector<ResultArgument> TuningManipulator::runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize)
{
    return manipulatorInterface->runKernel(kernelId, globalSize, localSize);
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

void TuningManipulator::updateArgumentVector(const size_t argumentId, const void* argumentData)
{
    manipulatorInterface->updateArgumentVector(argumentId, argumentData);
}

void TuningManipulator::updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t numberOfElements)
{
    manipulatorInterface->updateArgumentVector(argumentId, argumentData, numberOfElements);
}

void  TuningManipulator::setAutomaticArgumentUpdate(const bool flag)
{
    manipulatorInterface->setAutomaticArgumentUpdate(flag);
}

void TuningManipulator::updateKernelArguments(const size_t kernelId, const std::vector<size_t>& argumentIds)
{
    manipulatorInterface->updateKernelArguments(kernelId, argumentIds);
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

} // namespace ktt
