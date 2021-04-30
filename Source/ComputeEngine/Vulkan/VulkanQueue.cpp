#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanQueue.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>

namespace ktt
{

VulkanQueue::VulkanQueue(const VkQueue queue) :
    m_Queue(queue)
{}

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
