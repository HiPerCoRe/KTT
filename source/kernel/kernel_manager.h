#pragma once

#include <vector>

#include "kernel.h"
#include "kernel_configuration.h"

namespace ktt
{

class KernelManager
{
public:
    // Constructor
    KernelManager();

    // Core methods
    size_t addKernel(const std::string& kernelName, const std::string& source, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filename, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    std::string getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration);

    // Kernel handling methods
    bool addParameter(const size_t id, const KernelParameter& parameter);
    void addArgumentInt(const size_t id, const std::vector<int>& data);
    void addArgumentFloat(const size_t id, const std::vector<float>& data);
    void addArgumentDouble(const size_t id, const std::vector<double>& data);
    bool useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    // Getters
    size_t getKernelCount() const;

private:
    // Attributes
    size_t kernelCount;
    std::vector<Kernel> kernels;

    // Helper methods
    Kernel getKernel(const size_t id) const;
    std::string loadFileToString(const std::string& filename) const;
};

} // namespace ktt
