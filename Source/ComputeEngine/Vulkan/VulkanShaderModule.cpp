#ifdef KTT_API_VULKAN

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <ComputeEngine/Vulkan/VulkanDevice.h>
#include <ComputeEngine/Vulkan/VulkanShaderModule.h>
#include <ComputeEngine/Vulkan/VulkanUtility.h>

namespace ktt
{

VulkanShaderModule::VulkanShaderModule(const ShadercCompiler& compiler, const VulkanDevice& device, const std::string& name,
    const std::string& source, const DimensionVector& localSize, const KernelConfiguration& configuration) :
    m_Name(name),
    m_Source(source),
    m_Device(device.GetDevice())
{
    m_SpirvSource = compiler.Compile(name, source, shaderc_compute_shader, localSize, configuration);

    const VkShaderModuleCreateInfo createInfo =
    {
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        m_SpirvSource.size() * sizeof(uint32_t),
        m_SpirvSource.data()
    };

    CheckError(vkCreateShaderModule(m_Device, &createInfo, nullptr, &m_Module), "vkCreateShaderModule");
}

VulkanShaderModule::~VulkanShaderModule()
{
    vkDestroyShaderModule(m_Device, m_Module, nullptr);
}

const std::string& VulkanShaderModule::GetName() const
{
    return m_Name;
}

const std::string& VulkanShaderModule::GetSource() const
{
    return m_Source;
}

VkShaderModule VulkanShaderModule::GetModule() const
{
    return m_Module;
}

} // namespace ktt

#endif // KTT_API_VULKAN
