#pragma once

#include <memory>
#include <vector>

#include "compute_api_drivers/opencl_core.h"
#include "kernel/kernel_manager.h"
#include "tuning_runner/tuning_runner.h"

namespace ktt
{

class TunerCore
{
public:
    // Constructor
    explicit TunerCore(const size_t platformIndex, const std::vector<size_t>& deviceIndices);

    // Kernel manager methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    std::string getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const;
    std::vector<KernelConfiguration> getKernelConfigurations(const size_t id) const;

    void addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
        const Dimension& modifierDimension);
    void addArgumentInt(const size_t id, const std::vector<int>& data, const ArgumentMemoryType& argumentMemoryType);
    void addArgumentFloat(const size_t id, const std::vector<float>& data, const ArgumentMemoryType& argumentMemoryType);
    void addArgumentDouble(const size_t id, const std::vector<double>& data, const ArgumentMemoryType& argumentMemoryType);
    void useSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);

    size_t getKernelCount() const;
    const std::shared_ptr<const Kernel> getKernel(const size_t id) const;

    // Compute API methods
    static void printComputeAPIInfo(std::ostream& outputTarget);
    static std::vector<Platform> getPlatformInfo();
    static std::vector<Device> getDeviceInfo(const size_t platformIndex);
    void setCompilerOptions(const std::string& options);

private:
    // Attributes
    std::unique_ptr<KernelManager> kernelManager;
    std::unique_ptr<TuningRunner> tuningRunner;
    std::unique_ptr<OpenCLCore> openCLCore;
};

} // namespace ktt
