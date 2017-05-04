#include "tuning_manipulator.h"

namespace ktt
{

TuningManipulator::~TuningManipulator() = default;

std::vector<size_t> TuningManipulator::getUtilizedKernelIds() const
{
    return std::vector<size_t>{};
}

std::vector<ResultArgument> TuningManipulator::runKernel(const size_t kernelId)
{
    return manipulatorInterface->runKernel(kernelId);
}

std::vector<ResultArgument> TuningManipulator::runKernel(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize)
{
    return manipulatorInterface->runKernel(kernelId, globalSize, localSize);
}

void TuningManipulator::updateArgumentScalar(const size_t argumentId, const void* argumentData)
{
    manipulatorInterface->updateArgumentScalar(argumentId, argumentData);
}

void TuningManipulator::updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t dataSizeInBytes)
{
    manipulatorInterface->updateArgumentVector(argumentId, argumentData, dataSizeInBytes);
}

void TuningManipulator::setManipulatorInterface(ManipulatorInterface* manipulatorInterface)
{
    this->manipulatorInterface = manipulatorInterface;
}

} // namespace ktt