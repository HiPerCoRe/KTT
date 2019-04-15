#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <api/parameter_pair.h>
#include <shaderc_ktt/shaderc.hpp>

namespace ktt
{

class ShadercCompiler
{
public:
    static ShadercCompiler& getCompiler()
    {
        static ShadercCompiler instance;
        return instance;
    }

    std::vector<uint32_t> compile(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
        const std::vector<size_t>& localSize, const std::vector<ParameterPair>& parameterPairs)
    {
        shaderc::CompileOptions options(defaultOptions);

        options.AddMacroDefinition("LOCAL_SIZE_X", std::to_string(localSize[0]));
        options.AddMacroDefinition("LOCAL_SIZE_Y", std::to_string(localSize[1]));
        options.AddMacroDefinition("LOCAL_SIZE_Z", std::to_string(localSize[2]));
        addParameterDefinitions(options, parameterPairs);

        std::string preprocessedSource = preprocessShader(name, source, kind, options);
        std::vector<uint32_t> compiledSource = compileShader(name, preprocessedSource, kind, options);
        return compiledSource;
    }

    ShadercCompiler(const ShadercCompiler&) = delete;
    ShadercCompiler(ShadercCompiler&&) = delete;
    void operator=(const ShadercCompiler&) = delete;
    void operator=(ShadercCompiler&&) = delete;

private:
    shaderc::Compiler compiler;
    shaderc::CompileOptions defaultOptions;

    ShadercCompiler()
    {
        defaultOptions.SetOptimizationLevel(shaderc_optimization_level_performance);
    }

    std::string preprocessShader(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
        const shaderc::CompileOptions& options)
    {
        shaderc::PreprocessedSourceCompilationResult result = compiler.PreprocessGlsl(source, kind, name.c_str(), options);

        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(std::string("Vulkan shader compiler error: ") + result.GetErrorMessage());
        }

        return std::string{result.cbegin(), result.cend()};
    }

    std::vector<uint32_t> compileShader(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
        const shaderc::CompileOptions& options)
    {
        shaderc::SpvCompilationResult binaryModule = compiler.CompileGlslToSpv(source, kind, name.c_str(), options);

        if (binaryModule.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(std::string("Vulkan shader compiler error: ") + binaryModule.GetErrorMessage());
        }

        return std::vector<uint32_t>{binaryModule.cbegin(), binaryModule.cend()};
    }

    std::string compileShaderToAssembly(const std::string& name, const std::string& source, const shaderc_shader_kind kind,
        const shaderc::CompileOptions& options)
    {
        shaderc::AssemblyCompilationResult assembly = compiler.CompileGlslToSpvAssembly(source, kind, name.c_str(), options);

        if (assembly.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(std::string("Vulkan shader compiler error: ") + assembly.GetErrorMessage());
        }

        return std::string{assembly.cbegin(), assembly.cend()};
    }

    static void addParameterDefinitions(shaderc::CompileOptions& options, const std::vector<ParameterPair>& parameterPairs)
    {
        for (const auto& pair : parameterPairs)
        {
            const std::string& name = pair.getName();
            if (name == "LOCAL_SIZE_X" || name == "LOCAL_SIZE_Y" || name == "LOCAL_SIZE_Z")
            {
                throw std::runtime_error(std::string("Vulkan shader compiler error: Shader parameter is using reserved name: ") + name);
            }

            if (pair.hasValueDouble())
            {
                options.AddMacroDefinition(name, std::to_string(pair.getValueDouble()));
            }
            else
            {
                options.AddMacroDefinition(name, std::to_string(pair.getValue()));
            }
        }
    }
};

} // namespace ktt
