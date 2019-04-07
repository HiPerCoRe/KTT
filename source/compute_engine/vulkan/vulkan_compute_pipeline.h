#pragma once

#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <compute_engine/vulkan/vulkan_buffer.h>
#include <compute_engine/vulkan/vulkan_descriptor_pool.h>
#include <compute_engine/vulkan/vulkan_descriptor_set_holder.h>
#include <compute_engine/vulkan/vulkan_utility.h>

namespace ktt
{

class VulkanComputePipeline
{
public:
    VulkanComputePipeline() :
        device(nullptr),
        pipeline(nullptr),
        pipelineLayout(nullptr),
        descriptorSetLayout(nullptr),
        shaderName(""),
        descriptorPool(nullptr),
        descriptorSets(nullptr)
    {}

    explicit VulkanComputePipeline(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkShaderModule shader,
        const std::string& shaderName) :
        device(device),
        descriptorSetLayout(descriptorSetLayout),
        shaderName(shaderName),
        descriptorPool(nullptr),
        descriptorSets(nullptr)
    {
        const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
        {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            nullptr,
            0,
            1,
            &descriptorSetLayout,
            0,
            nullptr
        };

        checkVulkanError(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");

        const VkPipelineShaderStageCreateInfo shaderStageCreateInfo =
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_COMPUTE_BIT,
            shader,
            shaderName.c_str(),
            nullptr
        };

        const VkComputePipelineCreateInfo pipelineCreateInfo =
        {
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            nullptr,
            0,
            shaderStageCreateInfo,
            pipelineLayout,
            nullptr,
            0
        };

        checkVulkanError(vkCreateComputePipelines(device, nullptr, 1, &pipelineCreateInfo, nullptr, &pipeline), "vkCreateComputePipelines");
    }

    ~VulkanComputePipeline()
    {
        if (pipeline != nullptr)
        {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        }
    }

    VkDevice getDevice() const
    {
        return device;
    }

    VkPipeline getPipeline() const
    {
        return pipeline;
    }

    VkPipelineLayout getPipelineLayout() const
    {
        return pipelineLayout;
    }

    const std::string& getShaderName() const
    {
        return shaderName;
    }

    void bindArguments(const std::vector<VulkanBuffer*>& buffers)
    {
        const uint32_t descriptorCount = static_cast<uint32_t>(buffers.size());

        if (descriptorPool == nullptr || descriptorPool->getDescriptorCount() != descriptorCount)
        {
            descriptorSets.reset(nullptr);
            descriptorPool.reset(nullptr);
            descriptorPool = std::make_unique<VulkanDescriptorPool>(device, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount);
            descriptorSets = std::make_unique<VulkanDescriptorSetHolder>(device, descriptorPool->getDescriptorPool(), descriptorSetLayout);
        }

        for (size_t i = 0; i < buffers.size(); ++i)
        {
            descriptorSets->bindBuffer(*buffers[i], VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, static_cast<uint32_t>(i));
        }
    }

    void recordDispatchShaderCommand(VkCommandBuffer commandBuffer, const std::vector<size_t>& globalSize, VkQueryPool queryPool)
    {
        const VkCommandBufferBeginInfo commandBufferBeginInfo =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr
        };

        std::vector<VkDescriptorSet> sets = descriptorSets->getDescriptorSets();

        checkVulkanError(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo), "vkBeginCommandBuffer");

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, static_cast<uint32_t>(sets.size()), sets.data(), 0,
            nullptr);

        vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 0);
        vkCmdDispatch(commandBuffer, static_cast<uint32_t>(globalSize[0]), static_cast<uint32_t>(globalSize[1]),
            static_cast<uint32_t>(globalSize[2]));
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1);

        checkVulkanError(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

private:
    VkDevice device;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    std::string shaderName;
    std::unique_ptr<VulkanDescriptorPool> descriptorPool;
    std::unique_ptr<VulkanDescriptorSetHolder> descriptorSets;
};

} // namespace ktt
