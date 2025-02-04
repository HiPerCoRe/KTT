#pragma once

#ifdef KTT_API_CUDA

#include <string>
#include <cuda.h>
#include <nvrtc.h>

#if defined(KTT_PROFILING_CUPTI)
#include <nvperf_cuda_host.h>
#endif // KTT_PROFILING_CUPTI

#if defined(KTT_POWER_USAGE_NVML)
#include <nvml.h>
#endif // KTT_POWER_USAGE_NVML

#include <Api/ExceptionReason.h>
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiKtt.h>

namespace ktt
{

std::string GetEnumName(const CUresult value);
std::string GetEnumName(const nvrtcResult value);
void CheckError(const CUresult value, const std::string& function, const std::string& info = "");
void CheckError(const nvrtcResult value, const std::string& function, const std::string& info = "",
    const ExceptionReason reason = ExceptionReason::General);

#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)
std::string GetEnumName(const CUptiResult value);
void CheckError(const CUptiResult value, const std::string& function, const std::string& info = "");
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI

#if defined(KTT_PROFILING_CUPTI)
std::string GetEnumName(const NVPA_Status value);
void CheckError(const NVPA_Status value, const std::string& function, const std::string& info = "");
#endif // KTT_PROFILING_CUPTI

#if defined(KTT_POWER_USAGE_NVML)
std::string GetEnumName(const nvmlReturn_t value);
void CheckError(const nvmlReturn_t value, const std::string& function, const std::string& info = "");
#endif // KTT_POWER_USAGE_NVML

} // namespace ktt

#endif // KTT_API_CUDA
