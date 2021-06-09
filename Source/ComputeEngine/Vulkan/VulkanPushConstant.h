#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include <KernelArgument/KernelArgument.h>

namespace ktt
{

class VulkanPushConstant
{
public:
    explicit VulkanPushConstant(const std::vector<KernelArgument*>& arguments);

    const VkPushConstantRange& GetRange() const;
    const std::vector<uint8_t>& GetData() const;
    bool IsValid() const;

private:
    VkPushConstantRange m_Range;
    std::vector<uint8_t> m_Data;
};

} // namespace ktt

#endif // KTT_API_VULKAN
