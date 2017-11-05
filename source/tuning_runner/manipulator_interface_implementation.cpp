#include <stdexcept>
#include <utility>
#include "manipulator_interface_implementation.h"
#include "utility/ktt_utility.h"
#include "utility/timer.h"

namespace ktt
{

ManipulatorInterfaceImplementation::ManipulatorInterfaceImplementation(ComputeEngine* computeEngine) :
    computeEngine(computeEngine),
    currentResult(KernelRunResult(0, 0)),
    currentConfiguration(KernelConfiguration(DimensionVector(), DimensionVector(), std::vector<ParameterPair>{}))
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

void ManipulatorInterfaceImplementation::runKernel(const KernelId id, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    Timer timer;
    timer.start();

    auto dataPointer = kernelData.find(id);
    if (dataPointer == kernelData.end())
    {
        throw std::runtime_error(std::string("Kernel with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    KernelRuntimeData kernelData = dataPointer->second;
    kernelData.setGlobalSize(globalSize);
    kernelData.setLocalSize(localSize);

    KernelRunResult result = computeEngine->runKernel(kernelData, getArgumentPointers(kernelData.getArgumentIds()),
        std::vector<ArgumentOutputDescriptor>{});
    currentResult = KernelRunResult(currentResult.getDuration() + result.getDuration(), currentResult.getOverhead());

    timer.stop();
    currentResult.increaseOverhead(timer.getElapsedTime());
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

    computeEngine->updateArgument(id, argumentData, argumentPointer->second->getDataSizeInBytes());
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

void ManipulatorInterfaceImplementation::getArgumentVector(const ArgumentId id, void* destination) const
{
    computeEngine->downloadArgument(id, destination);
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

void ManipulatorInterfaceImplementation::destroyArgumentBuffer(const ArgumentId id)
{
    Timer timer;
    timer.start();

    computeEngine->clearBuffer(id);

    timer.stop();
    currentResult.increaseOverhead(timer.getElapsedTime());
}

void ManipulatorInterfaceImplementation::addKernel(const KernelId id, const KernelRuntimeData& data)
{
    kernelData.insert(std::make_pair(id, data));
}

void ManipulatorInterfaceImplementation::setConfiguration(const KernelConfiguration& configuration)
{
    currentConfiguration = configuration;
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
        computeEngine->uploadArgument(*argument.second);
    }
}

void ManipulatorInterfaceImplementation::downloadBuffers(const std::vector<ArgumentOutputDescriptor>& output) const
{
    for (const auto& descriptor : output)
    {
        if (descriptor.getOutputSizeInBytes() == 0)
        {
            computeEngine->downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination());
        }
        else
        {
            computeEngine->downloadArgument(descriptor.getArgumentId(), descriptor.getOutputDestination(), descriptor.getOutputSizeInBytes());
        }
    }
}

void ManipulatorInterfaceImplementation::clearData()
{
    currentResult = KernelRunResult(0, 0);
    currentConfiguration = KernelConfiguration(DimensionVector(), DimensionVector(), std::vector<ParameterPair>{});
    kernelData.clear();
    vectorArguments.clear();
    nonVectorArguments.clear();
}

KernelRunResult ManipulatorInterfaceImplementation::getCurrentResult() const
{
    return currentResult;
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
    const ArgumentUploadType& uploadType)
{
    auto argumentPointer = nonVectorArguments.find(id);
    if (argumentPointer == nonVectorArguments.end())
    {
        throw std::runtime_error(std::string("Argument with following id is not present in tuning manipulator: ") + std::to_string(id));
    }

    if (argumentPointer->second.getUploadType() != uploadType)
    {
        throw std::runtime_error("Cannot convert between scalar and vector arguments");
    }

    auto updatedArgument = KernelArgument(id, argumentData, numberOfElements, argumentPointer->second.getDataType(),
        argumentPointer->second.getMemoryLocation(), argumentPointer->second.getAccessType(), uploadType, true);

    nonVectorArguments.erase(id);
    nonVectorArguments.insert(std::make_pair(id, updatedArgument));
}

} // namespace ktt
