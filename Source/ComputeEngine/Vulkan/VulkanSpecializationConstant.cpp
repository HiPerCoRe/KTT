#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanSpecializationConstant.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

VulkanSpecializationConstant::VulkanSpecializationConstant(const std::vector<KernelArgument*>& arguments)
{
    for (size_t i = 0; i < arguments.size(); ++i)
    {
        const auto& argument = *arguments[i];
        KttAssert(argument.GetMemoryType() == ArgumentMemoryType::Scalar || argument.GetMemoryType() == ArgumentMemoryType::Symbol,
            "Only scalar and symbol arguments can be used as specialization constants");

        VkSpecializationMapEntry entry =
        {
            static_cast<uint32_t>(i),
            static_cast<uint32_t>(m_Data.size()),
            argument.GetDataSize()
        };

        for (size_t j = 0; j < argument.GetDataSize(); ++j)
        {
            m_Data.push_back(argument.GetDataWithType<uint8_t>()[j]);
        }

        m_MapEntries.push_back(entry);
    }

    m_SpecializationInfo.mapEntryCount = static_cast<uint32_t>(m_MapEntries.size()),
    m_SpecializationInfo.pMapEntries = m_MapEntries.data();
    m_SpecializationInfo.dataSize = m_Data.size();
    m_SpecializationInfo.pData = m_Data.data();
}

const VkSpecializationInfo& VulkanSpecializationConstant::GetSpecializationInfo() const
{
    return m_SpecializationInfo;
}

} // namespace ktt

#endif // KTT_API_VULKAN
