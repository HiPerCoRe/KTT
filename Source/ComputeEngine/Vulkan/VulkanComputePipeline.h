#pragma once

#ifdef KTT_API_VULKAN

#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <ComputeEngine/Vulkan/VulkanBuffer.h>
#include <ComputeEngine/Vulkan/VulkanDescriptorSetLayout.h>
#include <ComputeEngine/Vulkan/VulkanDescriptorSets.h>
#include <ComputeEngine/Vulkan/VulkanShaderModule.h>
#include <KernelArgument/KernelArgument.h>

namespace ktt
{

class ShadercCompiler;
class VulkanCommandPool;
class VulkanComputeAction;
class VulkanDescriptorPool;
class VulkanDevice;
class VulkanQueryPool;
class VulkanQueue;

class VulkanComputePipeline : public std::enable_shared_from_this<VulkanComputePipeline>
{
public:
    explicit VulkanComputePipeline(const VulkanDevice& device, IdGenerator<ComputeActionId>& generator, const ShadercCompiler& compiler,
        const std::string& name, const std::string& source, const DimensionVector& localSize, const KernelConfiguration& configuration,
        const VulkanDescriptorPool& descriptorPool, const std::vector<KernelArgument*>& arguments);
    ~VulkanComputePipeline();

    VkPipeline GetPipeline() const;
    const std::string& GetName() const;

    void BindArguments(const std::vector<VulkanBuffer*>& buffers);
    std::unique_ptr<VulkanComputeAction> DispatchShader(const VulkanQueue& queue, const VulkanCommandPool& commandPool,
        VulkanQueryPool& queryPool, const DimensionVector& globalSize, const std::vector<KernelArgument*>& scalarArguments);

private:
    DimensionVector m_LocalSize;
    const VulkanDevice& m_Device;
    VkPipeline m_Pipeline;
    VkPipelineLayout m_PipelineLayout;
    std::unique_ptr<VulkanShaderModule> m_ShaderModule;
    std::unique_ptr<VulkanDescriptorSetLayout> m_SetLayout;
    std::unique_ptr<VulkanDescriptorSets> m_DescriptorSets;
    IdGenerator<ComputeActionId>& m_Generator;
};

} // namespace ktt

#endif // KTT_API_VULKAN
