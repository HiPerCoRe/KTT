#pragma once

#include <string>
#include <vector>

#include "compute_api_driver/compute_api_driver.h"
#include "dto/kernel_run_result.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

class VulkanCore : public ComputeApiDriver
{
public:
    // Constructor
    explicit VulkanCore(const size_t deviceIndex);

    // Platform and device retrieval methods
    void printComputeApiInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;
    DeviceInfo getCurrentDeviceInfo() const override;

    // Compiler options setup
    void setCompilerOptions(const std::string& options) override;

    // Argument handling methods
    void uploadArgument(const KernelArgument& kernelArgument) override;
    void updateArgument(const size_t argumentId, const void* data, const size_t dataSizeInBytes) override;
    KernelArgument downloadArgument(const size_t argumentId) const override;
    void clearBuffer(const size_t argumentId) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentMemoryType& argumentMemoryType) override;

    // High-level kernel execution methods
    KernelRunResult runKernel(const std::string& source, const std::string& kernelName, const std::vector<size_t>& globalSize,
        const std::vector<size_t>& localSize, const std::vector<const KernelArgument*>& argumentPointers) override;

private:
    // Attributes
    size_t deviceIndex;
    std::string compilerOptions;
};

} // namespace ktt
