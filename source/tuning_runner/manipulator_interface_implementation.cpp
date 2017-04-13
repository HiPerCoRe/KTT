#include "manipulator_interface_implementation.h"

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(ArgumentManager* argumentManager, OpenCLCore* openCLCore) :
    argumentManager(argumentManager),
    openCLCore(openCLCore)
{}

std::vector<ResultArgument> ManipulatorInterfaceImplementation::runKernel(const size_t kernelId)
{
    std::vector<ResultArgument> result;
    return result;
    // to do
}

std::vector<ResultArgument> ManipulatorInterfaceImplementation::runKernel(const size_t kernelId, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    std::vector<ResultArgument> result;
    return result;
    // to do
}

void ManipulatorInterfaceImplementation::updateArgumentScalar(const size_t argumentId, const void* argumentData)
{
    // to do
}

void ManipulatorInterfaceImplementation::updateArgumentVector(const size_t argumentId, const void* argumentData, const size_t dataSizeInBytes)
{
    // to do
}

} // namespace ktt
