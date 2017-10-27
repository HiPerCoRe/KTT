#pragma once

#include <cstddef>
#include <string>
#include <tuple>

namespace ktt
{

using ArgumentId = size_t;
using KernelId = size_t;
using ParameterPair = std::tuple<std::string, size_t>;
using TunerFlag = bool;

} // namespace ktt
