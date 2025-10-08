#pragma once

#include <json.hpp>

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Output/KernelResult.h>
#include <Output/TunerMetadata.h>

namespace ktt
{

using json = nlohmann::json;

template <typename T>
    struct as_T4 {
        const T& v;
        explicit as_T4(const T& vv) : v(vv) {}
    };

template <typename T>
    inline void to_json(nlohmann::json& j, const as_T4<std::vector<T>>& w) {
        j = json::array();
        for (const auto& elem : w.v) {
            json j_elem;
            to_json(j_elem, as_T4<T>(elem));
            j.push_back(j_elem);
        }
    }

void to_json(json& j, const as_T4<KernelResult>& result);
void to_json(json& j, const as_T4<KernelConfiguration>& configuration);
void to_json(json& j, const as_T4<KernelProfilingCounter>& counter);
void to_json(json& j, const as_T4<TunerMetadata>& metadata);

inline void to_json(json& j, const as_T4<ResultStatus>& w) {
    switch (w.v) {
        case ResultStatus::Ok: j = "correct"; break;
        case ResultStatus::ComputationFailed: j = "runtime"; break;
        case ResultStatus::ValidationFailed: j = "correctness"; break;
        case ResultStatus::CompilationFailed: j = "compile"; break;
        case ResultStatus::DeviceLimitsExceeded: j = "runtime"; break;
    }
}

inline void to_json(json& j, const as_T4<TimeUnit>& w) {
    switch (w.v) {
        case TimeUnit::Nanoseconds: j = "nanoseconds"; break;
        case TimeUnit::Microseconds: j = "microseconds"; break;
        case TimeUnit::Milliseconds: j = "milliseconds"; break;
        case TimeUnit::Seconds: j = "seconds"; break;
    }
}

NLOHMANN_JSON_SERIALIZE_ENUM(ComputeApi,
{
    {ComputeApi::OpenCL, "OpenCL"},
    {ComputeApi::CUDA, "CUDA"},
    {ComputeApi::Vulkan, "Vulkan"}
});
/*
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
    {ResultStatus::Ok, "correct"},
    {ResultStatus::ComputationFailed, "runtime"},
    {ResultStatus::ValidationFailed, "correctness"},
    {ResultStatus::CompilationFailed, "compile"},
    {ResultStatus::DeviceLimitsExceeded, "runtime"}
    // timeout is marked as ComputationFailed in KTT
    // constraints os marked as CompilationFailed in KTT
});

NLOHMANN_JSON_SERIALIZE_ENUM(ParameterValueType,
{
    {ParameterValueType::Int, "Int"},
    {ParameterValueType::UnsignedInt, "UnsignedInt"},
    {ParameterValueType::Double, "Double"},
    {ParameterValueType::Float, "Float"},
    {ParameterValueType::Bool, "Bool"},
    {ParameterValueType::String, "String"
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

void from_json(const json& j, ParameterPair& pair);

void to_json(json& j, const DimensionVector& vector);
void from_json(const json& j, DimensionVector& vector);

void from_json(const json& j, KernelConfiguration& configuration);

void to_json(json& j, const KernelProfilingCounter& counter);
void from_json(const json& j, KernelProfilingCounter& counter);

void to_json(json& j, const KernelCompilationData& data);
void from_json(const json& j, KernelCompilationData& data);

void to_json(json& j, const KernelProfilingData& data);
void from_json(const json& j, KernelProfilingData& data);

void to_json(json& j, const ComputationResult& result);
void from_json(const json& j, ComputationResult& result);

void from_json(const json& j, KernelResult& result);
*/
} // namespace ktt
