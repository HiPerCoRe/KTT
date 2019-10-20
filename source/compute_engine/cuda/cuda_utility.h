#pragma once

#ifdef KTT_PLATFORM_CUDA

#include <string>
#include <cuda.h>
#include <nvrtc.h>

#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)
#include <cupti.h>
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI

namespace ktt
{

std::string getCUDAEnumName(const CUresult value);
std::string getNvrtcEnumName(const nvrtcResult value);
void checkCUDAError(const CUresult value);
void checkCUDAError(const CUresult value, const std::string& message);
void checkCUDAError(const nvrtcResult value, const std::string& message);
float getEventCommandDuration(const CUevent start, const CUevent end);

#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)
std::string getCUPTIEnumName(const CUptiResult value);
void checkCUPTIError(const CUptiResult value, const std::string& message);
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
