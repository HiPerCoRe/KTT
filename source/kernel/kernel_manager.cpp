#include <fstream>
#include <sstream>

#include "kernel_manager.h"

namespace ktt
{

KernelManager::KernelManager() {}

size_t KernelManager::addKernel(const std::string& kernelName, const std::string& source, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    Kernel kernel(kernelName, source, globalSize, localSize);
    kernels.push_back(kernel);

    return ++kernelCount;
}

size_t KernelManager::addKernelFromFile(const std::string& filename, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    std::string source = loadFileToString(filename);
    return addKernel(kernelName, source, globalSize, localSize);
}

std::string KernelManager::getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration)
{
    Kernel& kernel = getKernel(id);
    std::string source = kernel.getSource();

    for (auto& parameterValue : kernelConfiguration.getParameterValues())
    {
        std::stringstream ss;
        ss << std::get<1>(parameterValue);
        source = std::string("#define ") + std::get<0>(parameterValue) + std::string(" ") + ss.str() + std::string("\n") + source;
    }

    return source;
}

bool KernelManager::addParameter(const size_t id, const KernelParameter& parameter)
{
    return getKernel(id).addParameter(parameter);
}

void KernelManager::addArgumentInt(const size_t id, const std::vector<int>& data)
{
    getKernel(id).addArgumentInt(data);
}

void KernelManager::addArgumentFloat(const size_t id, const std::vector<float>& data)
{
    getKernel(id).addArgumentFloat(data);
}

void KernelManager::addArgumentDouble(const size_t id, const std::vector<double>& data)
{
    getKernel(id).addArgumentDouble(data);
}

bool KernelManager::useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    return getKernel(id).useSearchMethod(searchMethod, searchArguments);
}

size_t KernelManager::getKernelCount() const
{
    return kernelCount;
}

Kernel KernelManager::getKernel(const size_t id) const
{
    if (id >= kernelCount)
    {
        throw std::runtime_error("Invalid kernel id");
    }
    return kernels.at(id);
}

std::string KernelManager::loadFileToString(const std::string& filename) const
{
    std::ifstream file(filename);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

} // namespace ktt
