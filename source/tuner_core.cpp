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

void TunerCore::addParameter(const size_t id, const KernelParameter& parameter)
{
    kernelManager->addParameter(id, parameter);
}

void TunerCore::addArgumentInt(const size_t id, const std::vector<int>& data)
{
    kernelManager->addArgumentInt(id, data);
}

void TunerCore::addArgumentFloat(const size_t id, const std::vector<float>& data)
{
    kernelManager->addArgumentFloat(id, data);
}

void TunerCore::addArgumentDouble(const size_t id, const std::vector<double>& data)
{
    kernelManager->addArgumentDouble(id, data);
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

std::vector<OpenCLPlatform> TunerCore::getOpenCLPlatforms() const
{
    return openCLCore->getOpenCLPlatforms();
}

std::vector<OpenCLDevice> TunerCore::getOpenCLDevices(const OpenCLPlatform& platform) const
{
    return openCLCore->getOpenCLDevices(platform);
}

void TunerCore::printOpenCLInfo(std::ostream& outputTarget)
{
    OpenCLCore::printOpenCLInfo(outputTarget);
}

} // namespace ktt
