#pragma once

#ifdef KTT_PLATFORM_CUDA

#include <string>
#include <cuda.h>
#include <nvrtc.h>
#ifdef KTT_PROFILING
#include <cupti.h>
#endif // KTT_PROFILING

namespace ktt
{

std::string getCUDAEnumName(const CUresult value);
std::string getNvrtcEnumName(const nvrtcResult value);
void checkCUDAError(const CUresult value);
void checkCUDAError(const CUresult value, const std::string& message);
void checkCUDAError(const nvrtcResult value, const std::string& message);
float getEventCommandDuration(const CUevent start, const CUevent end);

#ifdef KTT_PROFILING
std::string getCUPTIEnumName(const CUptiResult value);
void checkCUDAError(const CUptiResult value, const std::string& message);
#endif // KTT_PROFILING

} // namespace ktt

#endif // KTT_PLATFORM_CUDA
