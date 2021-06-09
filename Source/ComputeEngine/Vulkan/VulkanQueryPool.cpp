#ifdef KTT_API_VULKAN

#include <array>

#include <Api/KttException.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanQueryPool.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

VulkanQueryPool::VulkanQueryPool(const VulkanDevice& device, const uint32_t maxConcurrentQueries) :
    m_Device(device.GetDevice()),
    m_TimestampPeriod(static_cast<double>(device.GetPhysicalDevice().GetProperties().limits.timestampPeriod))
{
    Logger::LogDebug("Initializing Vulkan query pool");

    const VkQueryPoolCreateInfo poolInfo =
    {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        nullptr,
        0,
        VK_QUERY_TYPE_TIMESTAMP,
        maxConcurrentQueries * 2,
        0
    };
        
    CheckError(vkCreateQueryPool(m_Device, &poolInfo, nullptr, &m_Pool), "vkCreateQueryPool");

    for (uint32_t i = 0; i < maxConcurrentQueries * 2; i += 2)
    {
        m_FreeQueryIds.insert(i);
    }
}

VulkanQueryPool::~VulkanQueryPool()
{
    Logger::LogDebug("Releasing Vulkan query pool");
    vkDestroyQueryPool(m_Device, m_Pool, nullptr);
}

Nanoseconds VulkanQueryPool::GetOperationDuration(const uint32_t firstqueryId)
{
    m_FreeQueryIds.insert(firstqueryId);
    std::array<uint64_t, 2> timestamps;

    CheckError(vkGetQueryPoolResults(m_Device, m_Pool, firstqueryId, 2, timestamps.size() * sizeof(uint64_t), timestamps.data(), 0,
        VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT), "vkGetQueryPoolResults");

    const uint64_t difference = timestamps[1] - timestamps[0];
    return static_cast<Nanoseconds>(difference * m_TimestampPeriod);
}

std::pair<uint32_t, uint32_t> VulkanQueryPool::AssignQueryIds()
{
    if (m_FreeQueryIds.empty())
    {
        throw KttException("Vulkan query pool ran out of free queries");
    }

    uint32_t result = *m_FreeQueryIds.begin();
    m_FreeQueryIds.erase(result);
    return std::make_pair(result, result + 1);
}

VkQueryPool VulkanQueryPool::GetPool() const
{
    return m_Pool;
}

} // namespace ktt

#endif // KTT_API_VULKAN
