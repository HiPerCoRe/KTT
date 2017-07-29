#pragma once

#include <string>

#include "vulkan/vulkan.h"
#include "vulkan_utility.h"

namespace ktt
{

class VulkanInstance
{
public:
    void initialize()
    {
        const VkApplicationInfo applicationInfo =
        {
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            nullptr,
            "ktt_compute",
            0,
            "",
            0,
            VK_MAKE_VERSION(1, 0, 51)
        };

        const VkInstanceCreateInfo instanceCreateInfo =
        {
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            nullptr,
            0,
            &applicationInfo,
            0,
            nullptr,
            0,
            nullptr
        };

        checkVulkanError(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));
    }

    VkInstance getInstance() const
    {
        return instance;
    }

private:
    VkInstance instance;
};

} // namespace ktt
