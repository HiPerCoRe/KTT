#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include <ComputeEngine/Vulkan/ShadercCompiler.h>

namespace ktt
{

class DimensionVector;
class KernelConfiguration;
class VulkanDevice;

class VulkanShaderModule
{
public:
    explicit VulkanShaderModule(const ShadercCompiler& compiler, const VulkanDevice& device, const std::string& name,
        const std::string& source, const DimensionVector& localSize, const KernelConfiguration& configuration);
    ~VulkanShaderModule();

    const std::string& GetName() const;
    const std::string& GetSource() const;
    VkShaderModule GetModule() const;

private:
    std::string m_Name;
    std::string m_Source;
    std::vector<uint32_t> m_SpirvSource;
    VkDevice m_Device;
    VkShaderModule m_Module;
};

} // namespace ktt

#endif // KTT_API_VULKAN
