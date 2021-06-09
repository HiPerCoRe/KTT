#pragma once

#ifdef KTT_API_VULKAN

#include <cstdint>
#include <string>
#include <vector>
#include <shaderc/shaderc.hpp>

namespace ktt
{

class DimensionVector;
class KernelConfiguration;

class ShadercCompiler
{
public:
    ShadercCompiler();

    std::vector<uint32_t> Compile(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
        const DimensionVector& localSize, const KernelConfiguration& configuration) const;

private:
    shaderc::Compiler m_Compiler;
    shaderc::CompileOptions m_DefaultOptions;

    std::string PreprocessShader(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
        const shaderc::CompileOptions& options) const;
    std::vector<uint32_t> CompileShader(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
        const shaderc::CompileOptions& options) const;
    std::string CompileShaderToAssembly(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
        const shaderc::CompileOptions& options) const;

    static void AddParameterDefinitions(shaderc::CompileOptions& options, const KernelConfiguration& configuration);
};

} // namespace ktt

#endif // KTT_API_VULKAN
