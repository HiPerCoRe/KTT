#pragma once

#include <array>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanQueryPool
{
public:
    explicit VulkanQueryPool(VkDevice device, const float timestampPeriod) :
        device(device),
        timestampPeriod(timestampPeriod)
    {
        const VkQueryPoolCreateInfo queryPoolInfo =
        {
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            nullptr,
            0,
            VK_QUERY_TYPE_TIMESTAMP,
            2,
            0
        };
        
        checkVulkanError(vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPool), "vkCreateQueryPool");
    }

    ~VulkanQueryPool()
    {
        vkDestroyQueryPool(device, queryPool, nullptr);
    }

    uint64_t getResult() const
    {
        std::array<uint64_t, 2> timestamps;

        checkVulkanError(vkGetQueryPoolResults(device, queryPool, 0, 2, 2 * sizeof(uint64_t), timestamps.data(), 0,
            VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT), "vkGetQueryPoolResults");

        const uint64_t difference = timestamps[1] - timestamps[0];
        return static_cast<uint64_t>(difference * timestampPeriod);
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkQueryPool getQueryPool() const
    {
        return queryPool;
    }

    float getTimestampPeriod() const
    {
        return timestampPeriod;
    }

private:
    VkDevice device;
    VkQueryPool queryPool;
    float timestampPeriod;
};

} // namespace ktt
