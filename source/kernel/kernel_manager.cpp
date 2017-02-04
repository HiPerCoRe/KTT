#include <fstream>
#include <sstream>

#include "kernel_manager.h"

namespace ktt
{

KernelManager::KernelManager():
    kernelCount(static_cast<size_t>(0))
{}

size_t KernelManager::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    Kernel kernel(source, kernelName, globalSize, localSize);
    kernels.push_back(std::make_shared<Kernel>(kernel));

    return kernelCount++;
}

size_t KernelManager::addKernelFromFile(const std::string& filename, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    std::string source = loadFileToString(filename);
    return addKernel(source, kernelName, globalSize, localSize);
}

std::string KernelManager::getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration)
{
    std::string source = getKernel(id)->getSource();

    for (auto& parameterValue : kernelConfiguration.getParameterValues())
    {
        std::stringstream stream;
        stream << std::get<1>(parameterValue);
        source = std::string("#define ") + std::get<0>(parameterValue) + std::string(" ") + stream.str() + std::string("\n") + source;
    }

    return source;
}

size_t KernelManager::getKernelCount() const
{
    return kernelCount;
}

const std::shared_ptr<Kernel> KernelManager::getKernel(const size_t id)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error("Invalid kernel id: " + id);
    }
    return kernels.at(id);
}

std::string KernelManager::loadFileToString(const std::string& filename) const
{
    std::ifstream file(filename);
    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

} // namespace ktt
