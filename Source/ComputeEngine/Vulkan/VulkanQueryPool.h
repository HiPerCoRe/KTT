#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <set>
#include <utility>
#include <vulkan/vulkan.h>

#include <KttTypes.h>

namespace ktt
{

class VulkanDevice;

class VulkanQueryPool
{
public:
    explicit VulkanQueryPool(const VulkanDevice& device, const uint32_t maxConcurrentQueries = 10);
    ~VulkanQueryPool();

    Nanoseconds GetOperationDuration(const uint32_t firstQueryId);
    std::pair<uint32_t, uint32_t> AssignQueryIds();
    VkQueryPool GetPool() const;

private:
    std::set<uint32_t> m_FreeQueryIds;
    VkDevice m_Device;
    VkQueryPool m_Pool;
    double m_TimestampPeriod;
};

} // namespace ktt

#endif // KTT_API_VULKAN
