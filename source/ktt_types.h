/** @file ktt_types.h
  * @brief File containing definitions of KTT type aliases.
  */
#pragma once

#include <cstddef>
#include <string>
#include <tuple>

namespace ktt
{

/** @typedef ArgumentId
  * @brief Data type for referencing kernel arguments in KTT.
  */
using ArgumentId = size_t;

/** @typedef KernelId
  * @brief Data type for referencing kernels in KTT.
  */
using KernelId = size_t;

/** @typedef ParameterPair
  * @brief Data type for holding single value for one kernel parameter. Parameter name and the value can be accessed by using std::get function.
  */
using ParameterPair = std::tuple<std::string, size_t>;

} // namespace ktt
