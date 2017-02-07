#pragma once

#include <memory>

#include "kernel/kernel_manager.h"
#include "tuning_runner/tuning_runner.h"

namespace ktt
{

class TunerCore
{
public:
    // Constructor
    TunerCore();

    // Kernel manager wrapper methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filename, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    std::string getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const;
    std::vector<KernelConfiguration> getKernelConfigurations(const size_t id) const;

    void addParameter(const size_t id, const KernelParameter& parameter);
    void addArgumentInt(const size_t id, const std::vector<int>& data);
    void addArgumentFloat(const size_t id, const std::vector<float>& data);
    void addArgumentDouble(const size_t id, const std::vector<double>& data);
    void useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    size_t getKernelCount() const;
    const std::shared_ptr<const Kernel> getKernel(const size_t id) const;

private:
    // Attributes
    std::shared_ptr<KernelManager> kernelManager;
    std::shared_ptr<TuningRunner> tuningRunner;
};

} // namespace ktt
