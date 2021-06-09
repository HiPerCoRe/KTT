#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanCommandBuffers.h>
#include <ComputeEngine/Vulkan/VulkanCommandPool.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

VulkanCommandBuffers::VulkanCommandBuffers(const VulkanDevice& device, const VulkanCommandPool& commandPool, const uint32_t count) :
    VulkanCommandBuffers(device, commandPool, count, VK_COMMAND_BUFFER_LEVEL_PRIMARY)
{}

VulkanCommandBuffers::VulkanCommandBuffers(const VulkanDevice& device, const VulkanCommandPool& commandPool, const uint32_t count,
    const VkCommandBufferLevel level) :
    m_Device(device.GetDevice()),
    m_Pool(commandPool.GetPool()),
    m_Level(level)
{
    KttAssert(count > 0, "Command buffer count must be greater than zero");

    const VkCommandBufferAllocateInfo allocateInfo =
    {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        m_Pool,
        m_Level,
        count
    };

    m_Buffers.resize(static_cast<size_t>(count));
    CheckError(vkAllocateCommandBuffers(m_Device, &allocateInfo, m_Buffers.data()), "vkAllocateCommandBuffers");
}

VulkanCommandBuffers::~VulkanCommandBuffers()
{
    vkFreeCommandBuffers(m_Device, m_Pool, static_cast<uint32_t>(m_Buffers.size()), m_Buffers.data());
}

VkCommandBuffer VulkanCommandBuffers::GetBuffer() const
{
    KttAssert(!m_Buffers.empty(), "Command buffers structure must hold at least one buffer");
    return m_Buffers[0];
}

const std::vector<VkCommandBuffer>& VulkanCommandBuffers::GetBuffers() const
{
    return m_Buffers;
}

VkCommandBufferLevel VulkanCommandBuffers::GetLevel() const
{
    return m_Level;
}

} // namespace ktt

#endif // KTT_API_VULKAN
