#include <fstream>
#include <sstream>

#include "kernel_manager.h"

namespace ktt
{

KernelManager::KernelManager() {}

size_t KernelManager::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    Kernel kernel(source, kernelName, globalSize, localSize);
    kernels.push_back(kernel);

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
    std::string source = getKernel(id).getSource();

    for (auto& parameterValue : kernelConfiguration.getParameterValues())
    {
        std::stringstream stream;
        stream << std::get<1>(parameterValue);
        source = std::string("#define ") + std::get<0>(parameterValue) + std::string(" ") + stream.str() + std::string("\n") + source;
    }

    return source;
}

void KernelManager::addParameter(const size_t id, const KernelParameter& parameter)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error("Kernel id is invalid");
    }
    kernels.at(id).addParameter(parameter);
}

void KernelManager::addArgumentInt(const size_t id, const std::vector<int>& data)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error("Kernel id is invalid");
    }
    kernels.at(id).addArgumentInt(data);
}

void KernelManager::addArgumentFloat(const size_t id, const std::vector<float>& data)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error("Kernel id is invalid");
    }
    kernels.at(id).addArgumentFloat(data);
}

void KernelManager::addArgumentDouble(const size_t id, const std::vector<double>& data)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error("Kernel id is invalid");
    }
    kernels.at(id).addArgumentDouble(data);
}

void KernelManager::useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments)
{
    if (id >= kernelCount)
    {
        throw std::runtime_error("Kernel id is invalid");
    }
    kernels.at(id).useSearchMethod(searchMethod, searchArguments);
}

Kernel KernelManager::getKernel(const size_t id) const
{
    if (id >= kernelCount)
    {
        throw std::runtime_error("Kernel id is invalid");
    }
    return kernels.at(id);
}

size_t KernelManager::getKernelCount() const
{
    return kernelCount;
}

std::string KernelManager::loadFileToString(const std::string& filename) const
{
    std::ifstream file(filename);
    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

} // namespace ktt
