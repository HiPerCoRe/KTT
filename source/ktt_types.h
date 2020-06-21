/** @file ktt_types.h
  * Definitions of KTT type aliases and constants.
  */
#pragma once

#include <cstdint>
#include <limits>

namespace ktt
{

/** @typedef PlatformIndex
  * Data type for referencing platforms in KTT.
  */
using PlatformIndex = uint32_t;

/** @typedef DeviceIndex
  * Data type for referencing devices in KTT.
  */
using DeviceIndex = uint32_t;

/** @typedef QueueId
  * Data type for referencing compute API queues in KTT.
  */
using QueueId = uint32_t;

/** @typedef KernelId
  * Data type for referencing kernels in KTT.
  */
using KernelId = uint64_t;

/** @typedef ArgumentId
  * Data type for referencing kernel arguments in KTT.
  */
using ArgumentId = uint64_t;

/** @typedef EventId
  * Data type for referencing compute API events in KTT.
  */
using EventId = uint64_t;

/** @typedef BufferMemory
  * Data type for accessing compute API buffers in KTT.
  */
using BufferMemory = void*;

/** @typedef UserContext
  * Data type for providing user-created compute device context.
  */
using UserContext = void*;

/** @typedef UserQueue
  * Data type for providing user-created compute device queue.
  */
using UserQueue = void*;

/** Kernel id returned by kernel addition methods in case of an error.
  */
const KernelId InvalidKernelId = std::numeric_limits<KernelId>::max();

/** Argument id returned by argument addition methods in case of an error.
  */
const ArgumentId InvalidArgumentId = std::numeric_limits<ArgumentId>::max();

} // namespace ktt
