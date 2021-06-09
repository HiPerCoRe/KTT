#ifdef KTT_API_VULKAN

#include <string>

#include <ComputeEngine/Vulkan/VulkanQueue.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

VulkanQueue::VulkanQueue(const QueueId id, const VkQueue queue) :
    m_Queue(queue),
    m_Id(id)
{
    Logger::LogDebug("Initializing Vulkan queue with id " + std::to_string(id));
}

QueueId VulkanQueue::GetId() const
{
    return m_Id;
}

void VulkanQueue::SubmitCommand(VkCommandBuffer buffer) const
{
    SubmitCommand(buffer, nullptr);
}

void VulkanQueue::SubmitCommand(VkCommandBuffer buffer, VkFence fence) const
{
    SubmitCommands(std::vector<VkCommandBuffer>{buffer}, fence);
}

void VulkanQueue::SubmitCommands(const std::vector<VkCommandBuffer>& buffers) const
{
    SubmitCommands(buffers, nullptr);
}

void VulkanQueue::SubmitCommands(const std::vector<VkCommandBuffer>& buffers, VkFence fence) const
{
    const VkSubmitInfo submitInfo =
    {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        static_cast<uint32_t>(buffers.size()),
        buffers.data(),
        0,
        nullptr
    };

    CheckError(vkQueueSubmit(m_Queue, 1, &submitInfo, fence), "vkQueueSubmit");
}

void VulkanQueue::WaitIdle() const
{
    vkQueueWaitIdle(m_Queue);
}

} // namespace ktt

#endif // KTT_API_VULKAN
