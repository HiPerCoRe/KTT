#pragma once

#include <json.hpp>

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/KernelResult.h>
#include <Output/TunerMetadata.h>

namespace ktt
{

using json = nlohmann::json;

NLOHMANN_JSON_SERIALIZE_ENUM(ComputeApi,
{
    {ComputeApi::OpenCL, "OpenCL"},
    {ComputeApi::CUDA, "CUDA"},
    {ComputeApi::Vulkan, "Vulkan"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(GlobalSizeType,
{
    {GlobalSizeType::OpenCL, "OpenCL"},
    {GlobalSizeType::CUDA, "CUDA"},
    {GlobalSizeType::Vulkan, "Vulkan"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(TimeUnit,
{
    {TimeUnit::Nanoseconds, "Nanoseconds"},
    {TimeUnit::Microseconds, "Microseconds"},
    {TimeUnit::Milliseconds, "Milliseconds"},
    {TimeUnit::Seconds, "Seconds"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ResultStatus,
{
    {ResultStatus::Ok, "Ok"},
    {ResultStatus::ComputationFailed, "ComputationFailed"},
    {ResultStatus::ValidationFailed, "ValidationFailed"},
    {ResultStatus::CompilationFailed, "CompilationFailed"},
    {ResultStatus::DeviceLimitsExceeded, "DeviceLimitsExceeded"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ParameterValueType,
{
    {ParameterValueType::Int, "Int"},
    {ParameterValueType::UnsignedInt, "UnsignedInt"},
    {ParameterValueType::Double, "Double"},
    {ParameterValueType::Float, "Float"},
    {ParameterValueType::Bool, "Bool"},
    {ParameterValueType::String, "String"}
});

NLOHMANN_JSON_SERIALIZE_ENUM(ProfilingCounterType,
{
    {ProfilingCounterType::Int, "Int"},
    {ProfilingCounterType::UnsignedInt, "UnsignedInt"},
    {ProfilingCounterType::Double, "Double"},
    {ProfilingCounterType::Percent, "Percent"},
    {ProfilingCounterType::Throughput, "Throughput"},
    {ProfilingCounterType::UtilizationLevel, "UtilizationLevel"}
});

void to_json(json& j, const TunerMetadata& metadata);
void from_json(const json& j, TunerMetadata& metadata);

void to_json(json& j, const ParameterPair& pair);
void from_json(const json& j, ParameterPair& pair);

void to_json(json& j, const DimensionVector& vector);
void from_json(const json& j, DimensionVector& vector);

void to_json(json& j, const KernelConfiguration& configuration);
void from_json(const json& j, KernelConfiguration& configuration);

void to_json(json& j, const KernelProfilingCounter& counter);
void from_json(const json& j, KernelProfilingCounter& counter);

void to_json(json& j, const KernelCompilationData& data);
void from_json(const json& j, KernelCompilationData& data);

void to_json(json& j, const KernelProfilingData& data);
void from_json(const json& j, KernelProfilingData& data);

void to_json(json& j, const ComputationResult& result);
void from_json(const json& j, ComputationResult& result);

void to_json(json& j, const KernelResult& result);
void from_json(const json& j, KernelResult& result);

} // namespace ktt
