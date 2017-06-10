#pragma once

#ifdef PLATFORM_CUDA

#include <string>

#include "cuda.h"
#include "nvrtc.h"
#include "enum/argument_memory_type.h"

namespace ktt
{

std::string getCudaEnumName(const CUresult value);
std::string getNvrtcEnumName(const nvrtcResult value);
void checkCudaError(const CUresult value);
void checkCudaError(const CUresult value, const std::string& message);
void checkCudaError(const nvrtcResult value, const std::string& message);
float getKernelRunDuration(const CUevent start, const CUevent end);

} // namespace ktt

#endif // PLATFORM_CUDA
