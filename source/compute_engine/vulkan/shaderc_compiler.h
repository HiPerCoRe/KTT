#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
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

    std::vector<uint32_t> compile(const std::string& name, const std::string& source, const shaderc_shader_kind kind)
    {
        std::string preprocessedSource = preprocessShader(name, source, kind);
        std::vector<uint32_t> compiledSource = compileShader(name, preprocessedSource, kind);
        return compiledSource;
    }

    ShadercCompiler(const ShadercCompiler&) = delete;
    ShadercCompiler(ShadercCompiler&&) = delete;
    void operator=(const ShadercCompiler&) = delete;
    void operator=(ShadercCompiler&&) = delete;

private:
    shaderc::Compiler compiler;
    shaderc::CompileOptions compilerOptions;

    ShadercCompiler()
    {
        compilerOptions.SetOptimizationLevel(shaderc_optimization_level_performance);
    }

    std::string preprocessShader(const std::string& name, const std::string& source, const shaderc_shader_kind kind)
    {
        shaderc::PreprocessedSourceCompilationResult result = compiler.PreprocessGlsl(source, kind, name.c_str(), compilerOptions);

        if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(std::string("Vulkan shader compiler error: ") + result.GetErrorMessage());
        }

        return std::string{result.cbegin(), result.cend()};
    }

    std::vector<uint32_t> compileShader(const std::string& name, const std::string& source, const shaderc_shader_kind kind)
    {
        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(source, kind, name.c_str(), compilerOptions);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            throw std::runtime_error(std::string("Vulkan shader compiler error: ") + module.GetErrorMessage());
        }

        return std::vector<uint32_t>{module.cbegin(), module.cend()};
    }
};

} // namespace ktt
