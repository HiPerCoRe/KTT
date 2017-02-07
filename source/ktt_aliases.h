#pragma once

#include <string>
#include <tuple>

#include "enums/kernel_argument_type.h"

namespace ktt
{

using ArgumentIndex = std::tuple<size_t, KernelArgumentType, size_t>;
using DimensionVector = std::tuple<size_t, size_t, size_t>;
using ParameterValue = std::tuple<std::string, size_t>;

} // namespace ktt
