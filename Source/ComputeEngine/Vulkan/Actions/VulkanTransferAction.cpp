#ifdef KTT_API_VULKAN

#include <string>

#include <ComputeEngine/Vulkan/Actions/VulkanTransferAction.h>
#include <ComputeEngine/Vulkan/VulkanCommandPool.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanQueryPool.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

VulkanTransferAction::VulkanTransferAction(const TransferActionId id, const VulkanDevice* device, const VulkanCommandPool* commandPool,
    VulkanQueryPool* queryPool) :
    m_Id(id),
    m_QueryPool(queryPool),
    m_Duration(InvalidDuration),
    m_Overhead(0),
    m_FirstQueryId(0),
    m_SecondQueryId(0)
{
    Logger::LogDebug("Initializing Vulkan transfer action with id " + std::to_string(id));

    if (commandPool != nullptr)
    {
        KttAssert(device != nullptr, "Async actions require valid device");
        KttAssert(m_QueryPool != nullptr, "Async actions require valid query pool");
        m_CommandBuffers = commandPool->AllocateBuffers(1);
        m_Fence = std::make_unique<VulkanFence>(*device);

        const auto ids = m_QueryPool->AssignQueryIds();
        m_FirstQueryId = ids.first;
        m_SecondQueryId = ids.second;
    }
}

void VulkanTransferAction::SetDuration(const Nanoseconds duration)
{
    KttAssert(!IsAsync(), "Duration for async actions is handled by query pool");
    m_Duration = duration;
}

void VulkanTransferAction::IncreaseOverhead(const Nanoseconds overhead)
{
    m_Overhead += overhead;
}

void VulkanTransferAction::WaitForFinish()
{
    Logger::LogDebug("Waiting for Vulkan transfer action with id " + std::to_string(m_Id));

    if (IsAsync())
    {
        m_Fence->Wait();
    }
}

TransferActionId VulkanTransferAction::GetId() const
{
    return m_Id;
}

VkFence VulkanTransferAction::GetFence() const
{
    KttAssert(IsAsync(), "Only async actions contain valid fence");
    return m_Fence->GetFence();
}

VkCommandBuffer VulkanTransferAction::GetCommandBuffer() const
{
    KttAssert(IsAsync(), "Only async actions contain valid command buffer");
    return m_CommandBuffers->GetBuffer();
}

uint32_t VulkanTransferAction::GetFirstQueryId() const
{
    KttAssert(IsAsync(), "Only async actions contain valid query ids");
    return m_FirstQueryId;
}

uint32_t VulkanTransferAction::GetSecondQueryId() const
{
    KttAssert(IsAsync(), "Only async actions contain valid query ids");
    return m_SecondQueryId;
}

Nanoseconds VulkanTransferAction::GetDuration() const
{
    if (IsAsync())
    {
        return m_QueryPool->GetOperationDuration(m_FirstQueryId);
    }

    return m_Duration;
}

Nanoseconds VulkanTransferAction::GetOverhead() const
{
    return m_Overhead;
}

bool VulkanTransferAction::IsAsync() const
{
    return m_CommandBuffers != nullptr;
}

TransferResult VulkanTransferAction::GenerateResult() const
{
    const Nanoseconds duration = GetDuration();
    const Nanoseconds overhead = GetOverhead();
    return TransferResult(duration, overhead);
}

} // namespace ktt

#endif // KTT_API_VULKAN
