#include "tuner_core.h"

namespace ktt
{

TunerCore::TunerCore(const size_t platformIndex, const std::vector<size_t>& deviceIndices):
    kernelManager(std::make_unique<KernelManager>()),
    tuningRunner(std::make_unique<TuningRunner>()),
    openCLCore(std::make_unique<OpenCLCore>(platformIndex, deviceIndices))
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

void TunerCore::addArgumentInt(const size_t id, const std::vector<int>& data, const ArgumentMemoryType& argumentMemoryType)
{
    kernelManager->addArgumentInt(id, data, argumentMemoryType);
}

void TunerCore::addArgumentFloat(const size_t id, const std::vector<float>& data, const ArgumentMemoryType& argumentMemoryType)
{
    kernelManager->addArgumentFloat(id, data, argumentMemoryType);
}

void TunerCore::addArgumentDouble(const size_t id, const std::vector<double>& data, const ArgumentMemoryType& argumentMemoryType)
{
    kernelManager->addArgumentDouble(id, data, argumentMemoryType);
}

void TunerCore::useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    kernelManager->useSearchMethod(id, searchMethod, searchArguments);
}

size_t TunerCore::getKernelCount() const
{
    return kernelManager->getKernelCount();
}

const std::shared_ptr<const Kernel> TunerCore::getKernel(const size_t id) const
{
    return kernelManager->getKernel(id);
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
