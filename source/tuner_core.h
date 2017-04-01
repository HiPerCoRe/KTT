#pragma once

#include <memory>
#include <vector>

#include "compute_api_drivers/opencl_core.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "tuning_runner/tuning_runner.h"

namespace ktt
{

class TunerCore
{
public:
    // Constructor
    explicit TunerCore(const size_t platformIndex, const size_t deviceIndex);

    // Kernel manager methods
    size_t addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize, const DimensionVector& localSize);
    size_t addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    std::string getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const;
    std::vector<KernelConfiguration> getKernelConfigurations(const size_t id) const;
    void addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
        const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    void setKernelArguments(const size_t id, const std::vector<size_t>& argumentIndices);
    void setSearchMethod(const size_t id, const SearchMethod& searchMethod, const std::vector<double>& searchArguments);
    size_t getKernelCount() const;
    const Kernel getKernel(const size_t id) const;

    // Argument manager methods
    template <typename T> size_t addArgument(const std::vector<T>& data, const ArgumentMemoryType& argumentMemoryType,
        const ArgumentQuantity& argumentQuantity)
    {
        return argumentManager->addArgument(data, argumentMemoryType, argumentQuantity);
    }

    template <typename T> void updateArgument(const size_t id, const std::vector<T>& data, const ArgumentQuantity& argumentQuantity)
    {
        argumentManager->updateArgument(id, data, argumentQuantity);
    }

    // Tuning runner methods
    std::vector<TuningResult> tuneKernel(const size_t id);

    // Compute API methods
    static void printComputeAPIInfo(std::ostream& outputTarget);
    static PlatformInfo getPlatformInfo(const size_t platformIndex);
    static std::vector<PlatformInfo> getPlatformInfoAll();
    static DeviceInfo getDeviceInfo(const size_t platformIndex, const size_t deviceIndex);
    static std::vector<DeviceInfo> getDeviceInfoAll(const size_t platformIndex);
    void setCompilerOptions(const std::string& options);

private:
    // Attributes
    std::unique_ptr<ArgumentManager> argumentManager;
    std::unique_ptr<KernelManager> kernelManager;
    std::unique_ptr<OpenCLCore> openCLCore;
    std::unique_ptr<TuningRunner> tuningRunner;
};

} // namespace ktt
