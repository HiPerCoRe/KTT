#pragma once

#include <string>

#include "cuda.h"

namespace ktt
{

void checkCudaError(const CUresult value);
void checkCudaError(const CUresult value, const std::string& message);

} // namespace ktt
