#pragma once

#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanSemaphore
{
public:
    VulkanSemaphore() :
        device(nullptr),
        semaphore(nullptr)
    {}

    explicit VulkanSemaphore(VkDevice device) :
        device(device)
    {
        const VkSemaphoreCreateInfo semaphoreCreateInfo =
        {
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            nullptr,
            0
        };

        checkVulkanError(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore), "vkCreateSemaphore");
    }

    ~VulkanSemaphore()
    {
        if (semaphore != nullptr)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkSemaphore getSemaphore() const
    {
        return semaphore;
    }

private:
    VkDevice device;
    VkSemaphore semaphore;
};

} // namespace ktt
