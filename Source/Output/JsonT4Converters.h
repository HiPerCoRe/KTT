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
        T& v;
        explicit as_T4(T& vv) : v(vv) {}
    };

template <typename T>
    inline void to_json(nlohmann::json& j, const as_T4<const std::vector<T>>& w) {
        j = json::array();
        for (const auto& elem : w.v) {
            json j_elem;
            to_json(j_elem, as_T4<const T>(elem));
            j.push_back(j_elem);
        }
    }

void to_json(json& j, const as_T4<const KernelResult>& result);
void to_json(json& j, const as_T4<const KernelConfiguration>& configuration);
void to_json(json& j, const as_T4<const KernelProfilingCounter>& counter);
void to_json(json& j, const as_T4<const TunerMetadata>& metadata);

template <typename T>
	inline void from_json(const nlohmann::json& j, as_T4<std::vector<T>>& w) {
		w.v.clear();
		w.v.reserve(j.size());

		for (const auto& j_elem : j) {
			T elem;
            auto elemWrapper = as_T4(elem);
			from_json(j_elem, elemWrapper);
			w.v.push_back(std::move(elem));
		}
	}


void from_json(const json& j, as_T4<KernelResult>& result);
void from_json(const json& j, as_T4<KernelConfiguration>& configuration);
void from_json(const json& j, as_T4<KernelProfilingCounter>& counter);
void from_json(const json& j, as_T4<TunerMetadata>& metadata);

inline void to_json(json& j, const as_T4<const ResultStatus>& w) {
    switch (w.v) {
        case ResultStatus::Ok: j = "correct"; break;
        case ResultStatus::ComputationFailed: j = "runtime"; break;
        case ResultStatus::ValidationFailed: j = "correctness"; break;
        case ResultStatus::CompilationFailed: j = "compile"; break;
        case ResultStatus::DeviceLimitsExceeded: j = "runtime"; break;
    }
}

inline void from_json(const json& j, as_T4<ResultStatus>& w) {
    const std::string& s = j.get_ref<const std::string&>();
    if (s == "correct") {
        w.v = ResultStatus::Ok;
    } else if (s == "runtime") {
        w.v = ResultStatus::ComputationFailed;
    } else if (s == "correctness") {
        w.v = ResultStatus::ValidationFailed;
    } else if (s == "compile") {
        w.v = ResultStatus::CompilationFailed;
    } else {
        throw KttException("During deserialization of json file, unknown ResultStatus string was detected: " + s);
    }
}


inline void to_json(json& j, const as_T4<const TimeUnit>& w) {
    switch (w.v) {
        case TimeUnit::Nanoseconds: j = "nanoseconds"; break;
        case TimeUnit::Microseconds: j = "microseconds"; break;
        case TimeUnit::Milliseconds: j = "milliseconds"; break;
        case TimeUnit::Seconds: j = "seconds"; break;
    }
}

inline void from_json(const json& j, as_T4<TimeUnit>& w) {
    const std::string& s = j.get_ref<const std::string&>();
    if (s == "nanoseconds") {
        w.v = TimeUnit::Nanoseconds;
    } else if (s == "microseconds") {
        w.v = TimeUnit::Microseconds;
    } else if (s == "milliseconds") {
        w.v = TimeUnit::Milliseconds;
    } else if (s == "seconds") {
        w.v = TimeUnit::Seconds;
    } else {
        throw KttException("During deserialization of json file, unknown TimeUnit string was detected: " + s);
    }
}


NLOHMANN_JSON_SERIALIZE_ENUM(ComputeApi,
{
    {ComputeApi::OpenCL, "OpenCL"},
    {ComputeApi::CUDA, "CUDA"},
    {ComputeApi::Vulkan, "Vulkan"}
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

} // namespace ktt
