#ifdef KTT_API_VULKAN

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <Api/KttException.h>
#include <ComputeEngine/Vulkan/ShadercCompiler.h>

namespace ktt
{

ShadercCompiler::ShadercCompiler()
{
    m_DefaultOptions.SetOptimizationLevel(shaderc_optimization_level_performance);
}

std::vector<uint32_t> ShadercCompiler::Compile(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
    const DimensionVector& localSize, const KernelConfiguration& configuration) const
{
    shaderc::CompileOptions options(m_DefaultOptions);

    options.AddMacroDefinition("LOCAL_SIZE_X", std::to_string(localSize.GetSizeX()));
    options.AddMacroDefinition("LOCAL_SIZE_Y", std::to_string(localSize.GetSizeY()));
    options.AddMacroDefinition("LOCAL_SIZE_Z", std::to_string(localSize.GetSizeZ()));
    AddParameterDefinitions(options, configuration);

    const std::string preprocessedSource = PreprocessShader(name, source, kind, options);
    return CompileShader(name, preprocessedSource, kind, options);
}

std::string ShadercCompiler::PreprocessShader(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
    const shaderc::CompileOptions& options) const
{
    const shaderc::PreprocessedSourceCompilationResult result = m_Compiler.PreprocessGlsl(source, kind, name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        throw KttException("Vulkan shader compiler error: " + result.GetErrorMessage());
    }

    return std::string{result.cbegin(), result.cend()};
}

std::vector<uint32_t> ShadercCompiler::CompileShader(const std::string& name, const std::string& source,
    const shaderc_shader_kind kind, const shaderc::CompileOptions& options) const
{
    const shaderc::SpvCompilationResult binaryModule = m_Compiler.CompileGlslToSpv(source, kind, name.c_str(), options);

    if (binaryModule.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        throw KttException("Vulkan shader compiler error: " + binaryModule.GetErrorMessage());
    }

    return std::vector<uint32_t>{binaryModule.cbegin(), binaryModule.cend()};
}

std::string ShadercCompiler::CompileShaderToAssembly(const std::string& name, const std::string& source,
    const shaderc_shader_kind kind, const shaderc::CompileOptions& options) const
{
    const shaderc::AssemblyCompilationResult assembly = m_Compiler.CompileGlslToSpvAssembly(source, kind, name.c_str(), options);

    if (assembly.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        throw KttException("Vulkan shader compiler error: " + assembly.GetErrorMessage());
    }

    return std::string{assembly.cbegin(), assembly.cend()};
}

void ShadercCompiler::AddParameterDefinitions(shaderc::CompileOptions& options, const KernelConfiguration& configuration)
{
    for (const auto& pair : configuration.GetPairs())
    {
        const std::string& name = pair.GetName();

        if (name == "LOCAL_SIZE_X" || name == "LOCAL_SIZE_Y" || name == "LOCAL_SIZE_Z")
        {
            throw KttException("Vulkan shader compiler error: Shader parameter is using reserved name " + name);
        }

        options.AddMacroDefinition(name, pair.GetValueString());
    }
}

} // namespace ktt

#endif // KTT_API_VULKAN
