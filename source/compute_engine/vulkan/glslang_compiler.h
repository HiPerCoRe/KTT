#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include "glslang/Public/ShaderLang.h"
#include "Standalone/DirStackFileIncluder.h"
#include "SPIRV/GlslangToSpv.h"

namespace ktt
{

class GlslangCompiler
{
public:
    static GlslangCompiler& getCompiler()
    {
        static GlslangCompiler instance;
        return instance;
    }

    std::vector<uint32_t> compile(const std::string& source, const EShLanguage type)
    {
        glslang::TShader shader(type);
        const char* inputString = source.c_str();
        shader.setStrings(&inputString, 1);

        const int dialectVersion = 100;
        const glslang::EShTargetClientVersion clientVersion = glslang::EShTargetVulkan_1_0;
        const glslang::EShTargetLanguageVersion languageVersion = glslang::EShTargetSpv_1_0;

        shader.setEnvInput(glslang::EShSourceGlsl, type, glslang::EShClientVulkan, dialectVersion);
        shader.setEnvClient(glslang::EShClientVulkan, clientVersion);
        shader.setEnvTarget(glslang::EShTargetSpv, languageVersion);

        TBuiltInResource resource = getDefaultResource();
        EShMessages message = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
        const int defaultVersion = 100;

        DirStackFileIncluder includer;
        std::string path = "./";
        includer.pushExternalLocalDirectory(path);

        std::string preprocessedGLSL;

        if (!shader.preprocess(&resource, defaultVersion, ENoProfile, false, false, message, &preprocessedGLSL, includer))
        {
            throw std::runtime_error(std::string("GLSL preprocessing failed: ") + shader.getInfoLog() + "\n" + shader.getInfoDebugLog());
        }

        const char* preprocessedCStr = preprocessedGLSL.c_str();
        shader.setStrings(&preprocessedCStr, 1);

        if (!shader.parse(&resource, defaultVersion, false, message))
        {
            throw std::runtime_error(std::string("GLSL parsing failed: ") + shader.getInfoLog() + "\n" + shader.getInfoDebugLog());
        }

        glslang::TProgram program;
        program.addShader(&shader);

        if (!program.link(message))
        {
            throw std::runtime_error(std::string("GLSL linking failed: ") + program.getInfoLog() + "\n" + program.getInfoDebugLog());
        }

        std::vector<uint32_t> spirvSource;
        spv::SpvBuildLogger logger;
        glslang::SpvOptions spvOptions;
        glslang::GlslangToSpv(*program.getIntermediate(type), spirvSource, &logger, &spvOptions);

        return spirvSource;
    }

    GlslangCompiler(const GlslangCompiler&) = delete;
    GlslangCompiler(GlslangCompiler&&) = delete;
    void operator=(const GlslangCompiler&) = delete;
    void operator=(GlslangCompiler&&) = delete;

private:
    bool initialized;

    GlslangCompiler()
    {
        if (!glslang::InitializeProcess())
        {
            throw std::runtime_error("Glslang compiler initialization failed");
        }
    }

    ~GlslangCompiler()
    {
        glslang::FinalizeProcess();
    }

    static TBuiltInResource getDefaultResource()
    {
        TBuiltInResource result =
        {
            32,    // maxLights
            6,     // maxClipPlanes
            32,    // maxTextureUnits
            32,    // maxTextureCoords
            64,    // maxVertexAttribs
            4096,  // maxVertexUniformComponents
            64,    // maxVaryingFloats
            32,    // maxVertexTextureImageUnits
            80,    // maxCombinedTextureImageUnits
            32,    // maxTextureImageUnits
            4096,  // maxFragmentUniformComponents
            32,    // maxDrawBuffers
            128,   // maxVertexUniformVectors
            8,     // maxVaryingVectors
            16,    // maxFragmentUniformVectors
            16,    // maxVertexOutputVectors
            15,    // maxFragmentInputVectors
            -8,    // minProgramTexelOffset
            7,     // maxProgramTexelOffset
            8,     // maxClipDistances
            65535, // maxComputeWorkGroupCountX
            65535, // maxComputeWorkGroupCountY
            65535, // maxComputeWorkGroupCountZ
            1024,  // maxComputeWorkGroupSizeX
            1024,  // maxComputeWorkGroupSizeY
            64,    // maxComputeWorkGroupSizeZ
            1024,  // maxComputeUniformComponents
            16,    // maxComputeTextureImageUnits
            8,     // maxComputeImageUniforms
            8,     // maxComputeAtomicCounters
            1,     // maxComputeAtomicCounterBuffers
            60,    // maxVaryingComponents
            64,    // maxVertexOutputComponents
            64,    // maxGeometryInputComponents
            128,   // maxGeometryOutputComponents
            128,   // maxFragmentInputComponents
            8,     // maxImageUnits
            8,     // maxCombinedImageUnitsAndFragmentOutputs
            8,     // maxCombinedShaderOutputResources
            0,     // maxImageSamples
            0,     // maxVertexImageUniforms
            0,     // maxTessControlImageUniforms
            0,     // maxTessEvaluationImageUniforms
            0,     // maxGeometryImageUniforms
            8,     // maxFragmentImageUniforms
            8,     // maxCombinedImageUniforms
            16,    // maxGeometryTextureImageUnits
            256,   // maxGeometryOutputVertices
            1024,  // maxGeometryTotalOutputComponents
            1024,  // maxGeometryUniformComponents
            64,    // maxGeometryVaryingComponents
            128,   // maxTessControlInputComponents
            128,   // maxTessControlOutputComponents
            16,    // maxTessControlTextureImageUnits
            1024,  // maxTessControlUniformComponents
            4096,  // maxTessControlTotalOutputComponents
            128,   // maxTessEvaluationInputComponents
            128,   // maxTessEvaluationOutputComponents
            16,    // maxTessEvaluationTextureImageUnits
            1024,  // maxTessEvaluationUniformComponents
            120,   // maxTessPatchComponents
            32,    // maxPatchVertices
            64,    // maxTessGenLevel
            16,    // maxViewports
            0,     // maxVertexAtomicCounters
            0,     // maxTessControlAtomicCounters
            0,     // maxTessEvaluationAtomicCounters
            0,     // maxGeometryAtomicCounters
            8,     // maxFragmentAtomicCounters
            8,     // maxCombinedAtomicCounters
            1,     // maxAtomicCounterBindings
            0,     // maxVertexAtomicCounterBuffers
            0,     // maxTessControlAtomicCounterBuffers
            0,     // maxTessEvaluationAtomicCounterBuffers
            0,     // maxGeometryAtomicCounterBuffers
            1,     // maxFragmentAtomicCounterBuffers
            1,     // maxCombinedAtomicCounterBuffers
            16384, // maxAtomicCounterBufferSize
            4,     // maxTransformFeedbackBuffers
            64,    // maxTransformFeedbackInterleavedComponents
            8,     // maxCullDistances
            8,     // maxCombinedClipAndCullDistances
            4,     // maxSamples
            256,   // maxMeshOutputVerticesNV
            512,   // maxMeshOutputPrimitivesNV
            32,    // maxMeshWorkGroupSizeX_NV
            1,     // maxMeshWorkGroupSizeY_NV
            1,     // maxMeshWorkGroupSizeZ_NV
            32,    // maxTaskWorkGroupSizeX_NV
            1,     // maxTaskWorkGroupSizeY_NV
            1,     // maxTaskWorkGroupSizeZ_NV
            4,     // maxMeshViewCountNV
            {      // TLimits
                1, // nonInductiveForLoops
                1, // whileLoops
                1, // doWhileLoops
                1, // generalUniformIndexing
                1, // generalAttributeMatrixVectorIndexing
                1, // generalVaryingIndexing
                1, // generalSamplerIndexing
                1, // generalVariableIndexing
                1 // generalConstantMatrixVectorIndexing
            }
        };

        return result;
    }
};

} // namespace ktt
