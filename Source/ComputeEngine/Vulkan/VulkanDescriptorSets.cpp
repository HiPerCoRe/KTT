#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/VulkanBuffer.h>
#include <ComputeEngine/Vulkan/VulkanDescriptorPool.h>
#include <ComputeEngine/Vulkan/VulkanDescriptorSetLayout.h>
#include <ComputeEngine/Vulkan/VulkanDescriptorSets.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

VulkanDescriptorSets::VulkanDescriptorSets(const VulkanDevice& device, const VulkanDescriptorPool& descriptorPool,
    const std::vector<const VulkanDescriptorSetLayout*>& descriptorLayouts) :
    m_Device(device.GetDevice()),
    m_Pool(descriptorPool.GetPool())
{
    KttAssert(descriptorLayouts.size() > 0, "Descriptor count must be greater than zero");
    std::vector<VkDescriptorSetLayout> layouts;

    for (const auto* layout : descriptorLayouts)
    {
        layouts.push_back(layout->GetLayout());
    }

    const VkDescriptorSetAllocateInfo allocateInfo =
    {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr,
        m_Pool,
        static_cast<uint32_t>(layouts.size()),
        layouts.data()
    };

    m_Sets.resize(descriptorLayouts.size());
    CheckError(vkAllocateDescriptorSets(m_Device, &allocateInfo, m_Sets.data()), "vkAllocateDescriptorSets");
}

VulkanDescriptorSets::~VulkanDescriptorSets()
{
    vkFreeDescriptorSets(m_Device, m_Pool, static_cast<uint32_t>(m_Sets.size()), m_Sets.data());
}

VkDescriptorSet VulkanDescriptorSets::GetSet() const
{
    KttAssert(!m_Sets.empty(), "Descriptor sets structure must hold at least one set");
    return m_Sets[0];
}

const std::vector<VkDescriptorSet>& VulkanDescriptorSets::GetSets() const
{
    return m_Sets;
}

void VulkanDescriptorSets::BindBuffer(const VulkanBuffer& buffer, const VkDescriptorType descriptorType,
    const size_t setIndex, const uint32_t binding)
{
    KttAssert(m_Sets.size() > static_cast<size_t>(setIndex), "Descriptor set index is out of range");

    const VkDescriptorBufferInfo bufferInfo =
    {
        buffer.GetBuffer(),
        0,
        buffer.GetSize()
    };

    const VkWriteDescriptorSet descriptorWrite =
    {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        m_Sets[setIndex],
        binding,
        0,
        1,
        descriptorType,
        nullptr,
        &bufferInfo,
        nullptr
    };

    vkUpdateDescriptorSets(m_Device, 1, &descriptorWrite, 0, nullptr);
}

void VulkanDescriptorSets::BindBuffers(const std::vector<VulkanBuffer*>& buffers, const VkDescriptorType descriptorType,
    const size_t setIndex)
{
    KttAssert(m_Sets.size() > static_cast<size_t>(setIndex), "Descriptor set index is out of range");
    
    std::vector<VkDescriptorBufferInfo> bufferInfos(buffers.size());
    std::vector<VkWriteDescriptorSet> descriptorWrites(buffers.size());

    for (size_t i = 0; i < buffers.size(); ++i)
    {
        bufferInfos[i].buffer = buffers[i]->GetBuffer();
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = buffers[i]->GetSize();

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].pNext = nullptr;
        descriptorWrites[i].dstSet = m_Sets[setIndex];
        descriptorWrites[i].dstBinding = static_cast<uint32_t>(i);
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].descriptorType = descriptorType;
        descriptorWrites[i].pImageInfo = nullptr;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        descriptorWrites[i].pTexelBufferView = nullptr;
    }

    vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

} // namespace ktt

#endif // KTT_API_VULKAN
