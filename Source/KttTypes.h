/** @file KttTypes.h
  * Definitions of KTT type aliases and constants.
  */
#pragma once

#include <cstdint>
#include <limits>
#include <string>

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

/** @typedef Nanoseconds
  * Data type for retrieving elapsed time in KTT.
  */
using Nanoseconds = uint64_t;

/** @typedef KernelComputeId
  * Data type for referencing unique kernel configurations in compute engine.
  */
using KernelComputeId = std::string;

/** @typedef ComputeActionId
  * Data type for referencing kernel compute actions in KTT.
  */
using ComputeActionId = uint64_t;

/** @typedef TransferActionId
  * Data type for referencing buffer transfer actions in KTT.
  */
using TransferActionId = uint64_t;

/** @typedef UnifiedBufferMemory
  * Data type for accessing unified memory buffers in KTT.
  */
using UnifiedBufferMemory = void*;

/** @typedef ComputeContext
  * Data type for providing custom compute device context.
  */
using ComputeContext = void*;

/** @typedef ComputeQueue
  * Data type for providing custom compute device queues.
  */
using ComputeQueue = void*;

/** @typedef ComputeBuffer
  * Data type for providing custom compute device buffers.
  */
using ComputeBuffer = void*;

/** Kernel id returned by kernel addition methods in case of an error.
  */
inline const KernelId InvalidKernelId = std::numeric_limits<KernelId>::max();

/** Argument id returned by argument addition methods in case of an error.
  */
inline const ArgumentId InvalidArgumentId = std::numeric_limits<ArgumentId>::max();

} // namespace ktt
