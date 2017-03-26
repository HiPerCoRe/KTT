#include "tuner_core.h"

namespace ktt
{

TunerCore::TunerCore(const size_t platformIndex, const size_t deviceIndex):
    argumentManager(std::make_unique<ArgumentManager>()),
    kernelManager(std::make_unique<KernelManager>()),
    openCLCore(std::make_unique<OpenCLCore>(platformIndex, deviceIndex)),
    tuningRunner(std::make_unique<TuningRunner>(argumentManager.get(), kernelManager.get(), openCLCore.get()))
{}

size_t TunerCore::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return kernelManager->addKernel(source, kernelName, globalSize, localSize);
}

size_t TunerCore::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    return kernelManager->addKernelFromFile(filePath, kernelName, globalSize, localSize);
}

std::string TunerCore::getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const
{
    return kernelManager->getKernelSourceWithDefines(id, kernelConfiguration);
}

std::vector<KernelConfiguration> TunerCore::getKernelConfigurations(const size_t id) const
{
    return kernelManager->getKernelConfigurations(id);
}

void TunerCore::addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values,
    const ThreadModifierType& threadModifierType, const Dimension& modifierDimension)
{
    kernelManager->addParameter(id, name, values, threadModifierType, modifierDimension);
}

void TunerCore::setKernelArguments(const size_t id, const std::vector<size_t>& argumentIndices)
{
    for (const auto index : argumentIndices)
    {
        if (index >= argumentManager->getArgumentCount())
        {
            throw std::runtime_error(std::string("Invalid kernel argument id: " + std::to_string(index)));
        }
    }

    kernelManager->setArguments(id, argumentIndices);
}

void TunerCore::useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    kernelManager->useSearchMethod(id, searchMethod, searchArguments);
}

size_t TunerCore::getKernelCount() const
{
    return kernelManager->getKernelCount();
}

const Kernel TunerCore::getKernel(const size_t id) const
{
    return kernelManager->getKernel(id);
}

std::vector<TuningResult> TunerCore::tuneKernel(const size_t id)
{
    return tuningRunner->tuneKernel(id);;
}

void TunerCore::printComputeAPIInfo(std::ostream& outputTarget)
{
    OpenCLCore::printOpenCLInfo(outputTarget);
}

PlatformInfo TunerCore::getPlatformInfo(const size_t platformIndex)
{
    return OpenCLCore::getOpenCLPlatformInfo(platformIndex);
}

std::vector<PlatformInfo> TunerCore::getPlatformInfoAll()
{
    return OpenCLCore::getOpenCLPlatformInfoAll();
}

DeviceInfo TunerCore::getDeviceInfo(const size_t platformIndex, const size_t deviceIndex)
{
    return OpenCLCore::getOpenCLDeviceInfo(platformIndex, deviceIndex);
}

std::vector<DeviceInfo> TunerCore::getDeviceInfoAll(const size_t platformIndex)
{
    return OpenCLCore::getOpenCLDeviceInfoAll(platformIndex);
}

void TunerCore::setCompilerOptions(const std::string& options)
{
    openCLCore->setOpenCLCompilerOptions(options);
}

} // namespace ktt
