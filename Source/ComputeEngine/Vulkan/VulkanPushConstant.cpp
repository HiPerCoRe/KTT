#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanPushConstant.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

VulkanPushConstant::VulkanPushConstant(const std::vector<KernelArgument*>& arguments)
{
    m_Range.size = 0;
    m_Range.offset = 0;
    m_Range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    for (const auto* argument : arguments)
    {
        KttAssert(argument->GetMemoryType() == ArgumentMemoryType::Scalar, "Only scalar arguments can be used as push constants");
        const size_t dataSize = argument->GetDataSize();

        for (size_t i = 0; i < dataSize; ++i)
        {
            m_Data.push_back(argument->GetDataWithType<uint8_t>()[i]);
        }

        m_Range.size += static_cast<uint32_t>(dataSize);
    }

    uint32_t paddingSize = 0;

    if (m_Range.size % 4 != 0)
    {
        paddingSize = 4 - m_Range.size % 4;
    }

    for (uint32_t i = 0; i < paddingSize; ++i)
    {
        m_Data.push_back(0);
    }

    m_Range.size += paddingSize;
}

const VkPushConstantRange& VulkanPushConstant::GetRange() const
{
    return m_Range;
}

const std::vector<uint8_t>& VulkanPushConstant::GetData() const
{
    return m_Data;
}

bool VulkanPushConstant::IsValid() const
{
    return !m_Data.empty();
}

} // namespace ktt

#endif // KTT_API_VULKAN
