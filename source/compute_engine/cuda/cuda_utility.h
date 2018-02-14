#pragma once

#ifdef PLATFORM_CUDA

#include <string>
#include "cuda.h"
#include "nvrtc.h"

namespace ktt
{

std::string getCUDAEnumName(const CUresult value);
std::string getNvrtcEnumName(const nvrtcResult value);
void checkCUDAError(const CUresult value);
void checkCUDAError(const CUresult value, const std::string& message);
void checkCUDAError(const nvrtcResult value, const std::string& message);
float getEventCommandDuration(const CUevent start, const CUevent end);

} // namespace ktt

#endif // PLATFORM_CUDA
