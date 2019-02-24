#pragma once

#include <string>
#include <vector>
#include <vulkan/vulkan.h>
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
        shaderName("")
    {}

    explicit VulkanComputePipeline(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkShaderModule shader,
        const std::string& shaderName) :
        device(device),
        shaderName(shaderName)
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

    void recordDispatchShaderCommand(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, const std::vector<size_t>& globalSize)
    {
        const VkCommandBufferBeginInfo commandBufferBeginInfo =
        {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            nullptr,
            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            nullptr
        };

        checkVulkanError(vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo), "vkBeginCommandBuffer");

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDispatch(commandBuffer, static_cast<uint32_t>(globalSize[0]), static_cast<uint32_t>(globalSize[1]),
            static_cast<uint32_t>(globalSize[2]));

        checkVulkanError(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    }

private:
    VkDevice device;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    std::string shaderName;
};

} // namespace ktt
