#pragma once

#include <string>
#include <tuple>

namespace ktt
{

using DimensionVector = std::tuple<size_t, size_t, size_t>;
using ParameterValue = std::tuple<std::string, size_t>;

} // namespace ktt
