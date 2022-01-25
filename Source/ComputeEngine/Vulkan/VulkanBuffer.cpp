#ifdef KTT_API_VULKAN

#include <cstring>

#include <Api/KttException.h>
#include <ComputeEngine/Vulkan/VulkanBuffer.h>
#include <ComputeEngine/Vulkan/VulkanCommandBuffers.h>
#include <ComputeEngine/Vulkan/VulkanCommandPool.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanQueryPool.h>
#include <ComputeEngine/Vulkan/VulkanQueue.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/Timer.h>

namespace ktt
{

VulkanBuffer::VulkanBuffer(KernelArgument& kernelArgument, IdGenerator<TransferActionId>& generator, const VulkanDevice& device,
    const VulkanMemoryAllocator& allocator, const VkBufferUsageFlags bufferUsage, const VmaMemoryUsage memoryUsage) :
    m_Argument(kernelArgument),
    m_Generator(generator),
    m_Device(device),
    m_Allocator(allocator.GetAllocator()),
    m_BufferSize(static_cast<VkDeviceSize>(kernelArgument.GetDataSize()))
{
    Logger::LogDebug("Initializing Vulkan buffer with id " + std::to_string(m_Argument.GetId()));

    const VkBufferCreateInfo bufferInfo =
    {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0,
        m_BufferSize,
        bufferUsage,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr
    };

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = memoryUsage;

    CheckError(vmaCreateBuffer(m_Allocator, &bufferInfo, &allocationInfo, &m_Buffer, &m_Allocation, nullptr), "vmaCreateBuffer");
}

VulkanBuffer::~VulkanBuffer()
{
    Logger::LogDebug("Releasing Vulkan buffer with id " + std::to_string(m_Argument.GetId()));
    vmaDestroyBuffer(m_Allocator, m_Buffer, m_Allocation);
}

std::unique_ptr<VulkanTransferAction> VulkanBuffer::UploadData(const void* source, const VkDeviceSize dataSize)
{
    Logger::LogDebug("Uploading data into Vulkan buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to upload is larger than size of buffer");
    }

    Timer timer;
    timer.Start();

    void* data;
    CheckError(vmaMapMemory(m_Allocator, m_Allocation, &data), "vmaMapMemory");
    std::memcpy(data, source, static_cast<size_t>(dataSize));
    vmaUnmapMemory(m_Allocator, m_Allocation);

    timer.Stop();

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<VulkanTransferAction>(id, InvalidQueueId);
    action->SetDuration(timer.GetElapsedTime());
    return action;
}

std::unique_ptr<VulkanTransferAction> VulkanBuffer::DownloadData(void* target, const VkDeviceSize dataSize)
{
    Logger::LogDebug("Downloading data from Vulkan buffer with id " + std::to_string(m_Argument.GetId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to download is larger than size of buffer");
    }

    Timer timer;
    timer.Start();

    void* data;
    CheckError(vmaMapMemory(m_Allocator, m_Allocation, &data), "vmaMapMemory");
    std::memcpy(target, data, static_cast<size_t>(dataSize));
    vmaUnmapMemory(m_Allocator, m_Allocation);

    timer.Stop();

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<VulkanTransferAction>(id, InvalidQueueId);
    action->SetDuration(timer.GetElapsedTime());
    return action;
}

std::unique_ptr<VulkanTransferAction> VulkanBuffer::CopyData(const VulkanQueue& queue, const VulkanCommandPool& commandPool,
    VulkanQueryPool& queryPool, const VulkanBuffer& source, const VkDeviceSize dataSize)
{
    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<VulkanTransferAction>(id, queue.GetId(), &m_Device, &commandPool, &queryPool);

    return CopyDataInternal(queue, queryPool, source, dataSize, std::move(action));
}

std::unique_ptr<VulkanTransferAction> VulkanBuffer::CopyData(const VulkanQueue& queue, const VulkanCommandPool& commandPool,
    VulkanQueryPool& queryPool, std::unique_ptr<VulkanBuffer> stagingSource, const VkDeviceSize dataSize)
{
    const auto& source = *stagingSource;

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<VulkanTransferAction>(id, queue.GetId(), &m_Device, &commandPool, &queryPool, std::move(stagingSource));

    return CopyDataInternal(queue, queryPool, source, dataSize, std::move(action));
}

VkBuffer VulkanBuffer::GetBuffer() const
{
    return m_Buffer;
}

KernelArgument& VulkanBuffer::GetArgument() const
{
    return m_Argument;
}

ArgumentId VulkanBuffer::GetArgumentId() const
{
    return m_Argument.GetId();
}

ArgumentAccessType VulkanBuffer::GetAccessType() const
{
    return m_Argument.GetAccessType();
}

ArgumentMemoryLocation VulkanBuffer::GetMemoryLocation() const
{
    return m_Argument.GetMemoryLocation();
}

VkDeviceSize VulkanBuffer::GetSize() const
{
    return m_BufferSize;
}

std::unique_ptr<VulkanTransferAction> VulkanBuffer::CopyDataInternal(const VulkanQueue& queue, VulkanQueryPool& queryPool,
    const VulkanBuffer& source, const VkDeviceSize dataSize, std::unique_ptr<VulkanTransferAction> action)
{
    Logger::LogDebug("Copying data into Vulkan buffer with id " + std::to_string(m_Argument.GetId())
        + " from buffer with id " + std::to_string(source.GetArgumentId()));

    if (m_BufferSize < dataSize)
    {
        throw KttException("Size of data to copy is larger than size of target buffer");
    }

    if (source.GetSize() < dataSize)
    {
        throw KttException("Size of data to copy is larger than size of source buffer");
    }

    const VkCommandBufferBeginInfo beginInfo =
    {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr
    };

    const VkBufferCopy copyRegion =
    {
        0,
        0,
        dataSize
    };

    CheckError(vkBeginCommandBuffer(action->GetCommandBuffer(), &beginInfo), "vkBeginCommandBuffer");

    vkCmdResetQueryPool(action->GetCommandBuffer(), queryPool.GetPool(), action->GetFirstQueryId(), 2);
    vkCmdWriteTimestamp(action->GetCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool.GetPool(),
        action->GetFirstQueryId());
    vkCmdCopyBuffer(action->GetCommandBuffer(), source.GetBuffer(), m_Buffer, 1, &copyRegion);
    vkCmdWriteTimestamp(action->GetCommandBuffer(), VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool.GetPool(),
        action->GetSecondQueryId());

    CheckError(vkEndCommandBuffer(action->GetCommandBuffer()), "vkEndCommandBuffer");

    queue.SubmitCommand(action->GetCommandBuffer(), action->GetFence());
    return action;
}

} // namespace ktt

#endif // KTT_API_VULKAN
