#include "default_manipulator.h"

namespace ktt
{

void DefaultManipulator::launchComputation(const size_t kernelId, const DimensionVector& globalSize, const DimensionVector& localSize,
    const std::vector<ParameterValue>& parameterValues)
{
    manipulatorInterface->runKernel(kernelId);
}

} // namespace ktt
