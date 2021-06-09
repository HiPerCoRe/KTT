#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include <KernelArgument/KernelArgument.h>

namespace ktt
{

class VulkanSpecializationConstant
{
public:
    explicit VulkanSpecializationConstant(const std::vector<KernelArgument*>& arguments);
    const VkSpecializationInfo& GetSpecializationInfo() const;

private:
    VkSpecializationInfo m_SpecializationInfo;
    std::vector<VkSpecializationMapEntry> m_MapEntries;
    std::vector<uint8_t> m_Data;
};

} // namespace ktt

#endif // KTT_API_VULKAN
