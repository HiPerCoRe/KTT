#ifdef KTT_API_VULKAN

#include <ComputeEngine/Vulkan/Actions/VulkanComputeAction.h>
#include <ComputeEngine/Vulkan/VulkanCommandPool.h>
#include <ComputeEngine/Vulkan/VulkanComputePipeline.h>
#include <ComputeEngine/Vulkan/VulkanDescriptorPool.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanPushConstant.h>
#include <ComputeEngine/Vulkan/VulkanQueryPool.h>
#include <ComputeEngine/Vulkan/VulkanQueue.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

VulkanComputePipeline::VulkanComputePipeline(const VulkanDevice& device, IdGenerator<ComputeActionId>& generator,
    const ShadercCompiler& compiler, const std::string& name, const std::string& source, const DimensionVector& localSize,
    const KernelConfiguration& configuration, const VulkanDescriptorPool& descriptorPool,
    const std::vector<KernelArgument*>& arguments) :
    m_LocalSize(localSize),
    m_Device(device),
    m_ShaderModule(std::make_unique<VulkanShaderModule>(compiler, device, name, source, localSize, configuration)),
    m_Generator(generator)
{
    Logger::LogDebug("Initializing Vulkan compute pipeline with name " + name);
    std::vector<KernelArgument*> scalarArguments;
    uint32_t vectorArgumentCount = 0;

    for (auto* argument : arguments)
    {
        if (argument->GetMemoryType() == ArgumentMemoryType::Scalar)
        {
            scalarArguments.push_back(argument);
        }
        else if (argument->GetMemoryType() == ArgumentMemoryType::Vector)
        {
            ++vectorArgumentCount;
        }
    }

    m_SetLayout = std::make_unique<VulkanDescriptorSetLayout>(device, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vectorArgumentCount);
    m_DescriptorSets = descriptorPool.AllocateSets(std::vector<const VulkanDescriptorSetLayout*>{m_SetLayout.get()});

    VkDescriptorSetLayout setLayout = m_SetLayout->GetLayout();
    VulkanPushConstant pushConstant(scalarArguments);

    const VkPipelineLayoutCreateInfo layoutCreateInfo =
    {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        0,
        1,
        &setLayout,
        1,
        &pushConstant.GetRange()
    };

    CheckError(vkCreatePipelineLayout(m_Device.GetDevice(), &layoutCreateInfo, nullptr, &m_PipelineLayout),
        "vkCreatePipelineLayout");

    const VkPipelineShaderStageCreateInfo shaderCreateInfo =
    {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        m_ShaderModule->GetModule(),
        GetName().c_str(),
        nullptr
    };

    const VkComputePipelineCreateInfo pipelineCreateInfo =
    {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        nullptr,
        0,
        shaderCreateInfo,
        m_PipelineLayout,
        nullptr,
        0
    };

    CheckError(vkCreateComputePipelines(m_Device.GetDevice(), nullptr, 1, &pipelineCreateInfo, nullptr, &m_Pipeline),
        "vkCreateComputePipelines");
}

VulkanComputePipeline::~VulkanComputePipeline()
{
    Logger::LogDebug("Releasing Vulkan compute pipeline with name " + GetName());
    vkDestroyPipeline(m_Device.GetDevice(), m_Pipeline, nullptr);
    vkDestroyPipelineLayout(m_Device.GetDevice(), m_PipelineLayout, nullptr);
}

VkPipeline VulkanComputePipeline::GetPipeline() const
{
    return m_Pipeline;
}

const std::string& VulkanComputePipeline::GetName() const
{
    return m_ShaderModule->GetName();
}

void VulkanComputePipeline::BindArguments(const std::vector<VulkanBuffer*>& buffers)
{
    m_DescriptorSets->BindBuffers(buffers, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0);
}

std::unique_ptr<VulkanComputeAction> VulkanComputePipeline::DispatchShader(const VulkanQueue& queue,
    const VulkanCommandPool& commandPool, VulkanQueryPool& queryPool, const DimensionVector& globalSize,
    const std::vector<KernelArgument*>& scalarArguments)
{
    const VkCommandBufferBeginInfo beginInfo =
    {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr
    };

    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<VulkanComputeAction>(id, m_Device, commandPool, queryPool, shared_from_this(), globalSize,
        m_LocalSize);
    std::vector<VkDescriptorSet> sets = m_DescriptorSets->GetSets();
    VulkanPushConstant pushConstant(scalarArguments);
    VkCommandBuffer commandBuffer = action->GetCommandBuffer();

    CheckError(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0,
        static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

    if (pushConstant.GetRange().size > 0)
    {
        vkCmdPushConstants(action->GetCommandBuffer(), m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            pushConstant.GetRange().size, pushConstant.GetData().data());
    }

    vkCmdResetQueryPool(commandBuffer, queryPool.GetPool(), action->GetFirstQueryId(), 2);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool.GetPool(), action->GetFirstQueryId());
    vkCmdDispatch(commandBuffer, static_cast<uint32_t>(globalSize.GetSizeX()), static_cast<uint32_t>(globalSize.GetSizeY()),
        static_cast<uint32_t>(globalSize.GetSizeZ()));
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool.GetPool(), action->GetSecondQueryId());

    CheckError(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    Logger::LogDebug("Launching compute pipeline " + GetName() + " with compute action id " + std::to_string(id)
        + ", global thread size: " + globalSize.GetString() + ", local thread size: " + m_LocalSize.GetString());
    queue.SubmitCommand(commandBuffer, action->GetFence());
    return action;
}

} // namespace ktt

#endif // KTT_API_VULKAN
