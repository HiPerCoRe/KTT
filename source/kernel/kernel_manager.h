#pragma once

#include <memory>
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
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filename, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    std::string getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration);

    // Getters
    size_t getKernelCount() const;
    const std::shared_ptr<Kernel> getKernel(const size_t id);

private:
    // Attributes
    size_t kernelCount;
    std::vector<std::shared_ptr<Kernel>> kernels;

    // Helper methods
    std::string loadFileToString(const std::string& filename) const;
};

} // namespace ktt
