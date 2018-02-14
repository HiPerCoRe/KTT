/** @file ktt_types.h
  * Definitions of KTT type aliases.
  */
#pragma once

#include <cstdint>

namespace ktt
{

/** @typedef ArgumentId
  * Data type for referencing kernel arguments in KTT.
  */
using ArgumentId = uint64_t;

/** @typedef DeviceIndex
  * Data type for referencing devices in KTT.
  */
using DeviceIndex = uint32_t;

/** @typedef KernelId
  * Data type for referencing kernels in KTT.
  */
using KernelId = uint64_t;

/** @typedef PlatformIndex
  * Data type for referencing platforms in KTT.
  */
using PlatformIndex = uint32_t;

/** @typedef QueueId
  * Data type for referencing compute API queues in KTT.
  */
using QueueId = uint32_t;

/** @typedef EventId
  * Data type for referencing compute API events in KTT.
  */
using EventId = uint64_t;

} // namespace ktt
