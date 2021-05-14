#ifdef KTT_API_VULKAN

#include <string>

#include <ComputeEngine/Vulkan/Actions/VulkanComputeAction.h>
#include <ComputeEngine/Vulkan/Actions/VulkanComputeAction.h>
#include <ComputeEngine/Vulkan/Actions/VulkanComputeAction.h>
#include <ComputeEngine/Vulkan/VulkanCommandPool.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanQueryPool.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

VulkanComputeAction::VulkanComputeAction(const ComputeActionId id, const VulkanDevice& device, const VulkanCommandPool& commandPool,
    VulkanQueryPool& queryPool, std::shared_ptr<VulkanComputePipeline> pipeline, const DimensionVector& globalSize,
    const DimensionVector& localSize) :
    m_Id(id),
    m_Pipeline(pipeline),
    m_QueryPool(queryPool),
    m_Overhead(0),
    m_GlobalSize(globalSize),
    m_LocalSize(localSize)
{
    Logger::LogDebug("Initializing Vulkan compute action with id " + std::to_string(id) + " for kernel with name "
        + pipeline->GetName());
    KttAssert(m_Pipeline != nullptr, "Invalid kernel was used during Vulkan compute action initialization");

    m_Fence = std::make_unique<VulkanFence>(device);
    m_CommandBuffers = commandPool.AllocateBuffers(1);

    const auto ids = m_QueryPool.AssignQueryIds();
    m_FirstQueryId = ids.first;
    m_SecondQueryId = ids.second;
}

void VulkanComputeAction::IncreaseOverhead(const Nanoseconds overhead)
{
    m_Overhead += overhead;
}

void VulkanComputeAction::SetComputeId(const KernelComputeId& id)
{
    m_ComputeId = id;
}

void VulkanComputeAction::WaitForFinish()
{
    Logger::LogDebug("Waiting for Vulkan compute action with id " + std::to_string(m_Id));
    m_Fence->Wait();
}

ComputeActionId VulkanComputeAction::GetId() const
{
    return m_Id;
}

VulkanComputePipeline& VulkanComputeAction::GetPipeline()
{
    return *m_Pipeline;
}

VkFence VulkanComputeAction::GetFence() const
{
    return m_Fence->GetFence();
}

VkCommandBuffer VulkanComputeAction::GetCommandBuffer() const
{
    return m_CommandBuffers->GetBuffer();
}

uint32_t VulkanComputeAction::GetFirstQueryId() const
{
    return m_FirstQueryId;
}

uint32_t VulkanComputeAction::GetSecondQueryId() const
{
    return m_SecondQueryId;
}

Nanoseconds VulkanComputeAction::GetDuration() const
{
    return m_QueryPool.GetOperationDuration(m_FirstQueryId);
}

Nanoseconds VulkanComputeAction::GetOverhead() const
{
    return m_Overhead;
}

const KernelComputeId& VulkanComputeAction::GetComputeId() const
{
    return m_ComputeId;
}

ComputationResult VulkanComputeAction::GenerateResult() const
{
    ComputationResult result(m_Pipeline->GetName());
    const Nanoseconds duration = GetDuration();
    const Nanoseconds overhead = GetOverhead();

    // Todo: generate compilation data from pipeline
    std::unique_ptr<KernelCompilationData> compilationData = nullptr;

    result.SetDurationData(duration, overhead);
    result.SetSizeData(m_GlobalSize, m_LocalSize);
    result.SetCompilationData(std::move(compilationData));

    return result;
}

} // namespace ktt

#endif // KTT_API_VULKAN
