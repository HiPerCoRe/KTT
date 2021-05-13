#pragma once

#ifdef KTT_API_VULKAN

#include <memory>
#include <vulkan/vulkan.h>

#include <ComputeEngine/Vulkan/Actions/VulkanTransferAction.h>
#include <ComputeEngine/Vulkan/VulkanMemoryAllocator.h>
#include <KernelArgument/KernelArgument.h>
#include <Utility/IdGenerator.h>
#include <KttTypes.h>

namespace ktt
{

class VulkanCommandPool;
class VulkanDevice;
class VulkanQueryPool;
class VulkanQueue;

class VulkanBuffer
{
public:
    explicit VulkanBuffer(KernelArgument& kernelArgument, IdGenerator<TransferActionId>& generator, const VulkanDevice& device,
        const VulkanMemoryAllocator& allocator, const VkBufferUsageFlags bufferUsage, const VmaMemoryUsage memoryUsage);
    ~VulkanBuffer();

    std::unique_ptr<VulkanTransferAction> UploadData(const void* source, const VkDeviceSize dataSize);
    std::unique_ptr<VulkanTransferAction> DownloadData(void* target, const VkDeviceSize dataSize);
    std::unique_ptr<VulkanTransferAction> CopyData(const VulkanQueue& queue, const VulkanCommandPool& commandPool,
        VulkanQueryPool& queryPool, const VulkanBuffer& source, const VkDeviceSize dataSize);

    VkBuffer GetBuffer() const;
    ArgumentId GetArgumentId() const;
    ArgumentAccessType GetAccessType() const;
    ArgumentMemoryLocation GetMemoryLocation() const;
    VkDeviceSize GetSize() const;

private:
    KernelArgument& m_Argument;
    IdGenerator<TransferActionId>& m_Generator;
    const VulkanDevice& m_Device;
    VkBuffer m_Buffer;
    VmaAllocation m_Allocation;
    VmaAllocator m_Allocator;
    VkDeviceSize m_BufferSize;
};

} // namespace ktt

#endif // KTT_API_VULKAN
