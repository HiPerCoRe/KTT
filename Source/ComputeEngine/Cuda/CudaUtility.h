#pragma once

#ifdef KTT_API_CUDA

#include <string>
#include <cuda.h>
#include <nvrtc.h>

#include <ComputeEngine/Cuda/CuptiLegacy/CuptiKtt.h>

namespace ktt
{

std::string GetEnumName(const CUresult value);
std::string GetEnumName(const nvrtcResult value);
void CheckError(const CUresult value, const std::string& function, const std::string& info = "");
void CheckError(const nvrtcResult value, const std::string& function, const std::string& info = "");

#if defined(KTT_PROFILING_CUPTI_LEGACY) || defined(KTT_PROFILING_CUPTI)
std::string GetEnumName(const CUptiResult value);
void CheckError(const CUptiResult value, const std::string& function, const std::string& info = "");
#endif // KTT_PROFILING_CUPTI_LEGACY || KTT_PROFILING_CUPTI

} // namespace ktt

#endif // KTT_API_CUDA
