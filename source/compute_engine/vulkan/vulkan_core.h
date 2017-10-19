#pragma once

#include <memory>
#include <string>
#include <vector>

#ifdef PLATFORM_VULKAN
#include "vulkan/vulkan.h"

#include "vulkan_buffer.h"
#include "vulkan_command_buffer.h"
#include "vulkan_command_pool.h"
#include "vulkan_fence.h"
#include "vulkan_instance.h"
#include "vulkan_device.h"
#include "vulkan_physical_device.h"
#include "vulkan_queue.h"
#include "vulkan_shader_module.h"
#include "vulkan_utility.h"
#endif // PLATFORM_VULKAN

#include "compute_engine/compute_engine.h"
#include "dto/kernel_run_result.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

#ifdef PLATFORM_VULKAN

class VulkanCore : public ComputeEngine
{
public:
    // Constructor
    explicit VulkanCore(const size_t deviceIndex);

    // Platform and device retrieval methods
    void printComputeApiInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;
    DeviceInfo getCurrentDeviceInfo() const override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;

    // Argument handling methods
    void uploadArgument(KernelArgument& kernelArgument) override;
    void updateArgument(const size_t argumentId, const void* data, const size_t dataSizeInBytes) override;
    KernelArgument downloadArgument(const size_t argumentId) const override;
    void downloadArgument(const size_t argumentId, void* destination) const override;
    void downloadArgument(const size_t argumentId, void* destination, const size_t dataSizeInBytes) const override;
    void clearBuffer(const size_t argumentId) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType& accessType) override;

    // High-level kernel execution methods
    KernelRunResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;

private:
    // Attributes
    size_t deviceIndex;
    std::string compilerOptions;
    bool globalSizeCorrection;
    VulkanInstance vulkanInstance;
    std::unique_ptr<VulkanDevice> device;
    std::unique_ptr<VulkanQueue> queue;
    std::unique_ptr<VulkanCommandPool> commandPool;
    std::unique_ptr<VulkanCommandBuffer> commandBuffer;

    // Helper Methods
    DeviceInfo getVulkanDeviceInfo(const size_t deviceIndex) const;
    std::vector<VulkanPhysicalDevice> getVulkanDevices() const;
    static DeviceType getDeviceType(const VkPhysicalDeviceType deviceType);
};

#else

class VulkanCore : public ComputeEngine
{
public:
    // Constructor
    explicit VulkanCore(const size_t deviceIndex);

    // Platform and device retrieval methods
    void printComputeApiInfo(std::ostream& outputTarget) const override;
    std::vector<PlatformInfo> getPlatformInfo() const override;
    std::vector<DeviceInfo> getDeviceInfo(const size_t platformIndex) const override;
    DeviceInfo getCurrentDeviceInfo() const override;

    // Utility methods
    void setCompilerOptions(const std::string& options) override;
    void setAutomaticGlobalSizeCorrection(const bool flag) override;

    // Argument handling methods
    void uploadArgument(KernelArgument& kernelArgument) override;
    void updateArgument(const size_t argumentId, const void* data, const size_t dataSizeInBytes) override;
    KernelArgument downloadArgument(const size_t argumentId) const override;
    void downloadArgument(const size_t argumentId, void* destination) const override;
    void downloadArgument(const size_t argumentId, void* destination, const size_t dataSizeInBytes) const override;
    void clearBuffer(const size_t argumentId) override;
    void clearBuffers() override;
    void clearBuffers(const ArgumentAccessType& accessType) override;

    // High-level kernel execution methods
    KernelRunResult runKernel(const KernelRuntimeData& kernelData, const std::vector<KernelArgument*>& argumentPointers,
        const std::vector<ArgumentOutputDescriptor>& outputDescriptors) override;
};

#endif // PLATFORM_VULKAN

} // namespace ktt
