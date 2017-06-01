#pragma once

#ifdef PLATFORM_CUDA

#include <string>

#include "cuda.h"
#include "../../enum/argument_memory_type.h"

namespace ktt
{

std::string getCudaEnumName(const CUresult value);
void checkCudaError(const CUresult value);
void checkCudaError(const CUresult value, const std::string& message);
float getKernelRunDuration(const CUevent start, const CUevent end);

} // namespace ktt

#endif // PLATFORM_CUDA
