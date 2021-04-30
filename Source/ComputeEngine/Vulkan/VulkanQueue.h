#pragma once

#ifdef KTT_API_VULKAN

#include <vector>
#include <vulkan/vulkan.h>

namespace ktt
{

class VulkanQueue
{
public:
    explicit VulkanQueue(const VkQueue queue);

    void SubmitCommand(VkCommandBuffer buffer) const;
    void SubmitCommand(VkCommandBuffer buffer, VkFence fence) const;
    void SubmitCommands(const std::vector<VkCommandBuffer>& buffers) const;
    void SubmitCommands(const std::vector<VkCommandBuffer>& buffers, VkFence fence) const;
    void WaitIdle() const;

private:
    VkQueue m_Queue;
};

} // namespace ktt

#endif // KTT_API_VULKAN
