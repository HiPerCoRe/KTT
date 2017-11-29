/** @file ktt_types.h
  * @brief Definitions for KTT type aliases.
  */
#pragma once

#include <cstddef>

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

/** @typedef QueueId
  * @brief Data type for referencing compute API queues in KTT.
  */
using QueueId = size_t;

} // namespace ktt
