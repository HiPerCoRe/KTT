/** @file KttTypes.h
  * Definitions of KTT type aliases and constants.
  */
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ktt
{

class ComputeInterface;

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

/** @typedef KernelDefinitionId
  * Data type for referencing kernel definitions in KTT.
  */
using KernelDefinitionId = uint64_t;

/** @typedef KernelId
  * Data type for referencing kernels in KTT.
  */
using KernelId = uint64_t;

/** @typedef ArgumentId
  * Data type for referencing kernel arguments in KTT.
  */
using ArgumentId = uint64_t;

/** @typedef ParameterInput
  * Data type used for creating and retrieving specific kernel configurations from KTT.
  */
using ParameterInput = std::vector<std::pair<std::string, std::variant<uint64_t, double>>>;

/** @typedef UserData
  * Custom user data which can be saved or loaded together with tuning results.
  */
using UserData = std::map<std::string, std::string>;

/** @typedef Nanoseconds
  * Data type for storing elapsed times in KTT.
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

/** @typedef ModifierFunction
  * Definition of kernel thread modifier function.
  */
using ModifierFunction = std::function<uint64_t(const uint64_t /*defaultSize*/, const std::vector<uint64_t>& /*parameterValues*/)>;

/** @typedef ConstraintFunction
  * Definition of kernel constraint function.
  */
using ConstraintFunction = std::function<bool(const std::vector<uint64_t>& /*parameterValues*/)>;

/** @typedef KernelLauncher
  * Definition of kernel launch function.
  */
using KernelLauncher = std::function<void(ComputeInterface& /*interface*/)>;

/** @typedef ReferenceComputation
  * Function for computing reference kernel argument output. Used during validation.
  */
using ReferenceComputation = std::function<void(void* /*buffer*/)>;

/** @typedef ValueComparator
  * Custom comparison function for kernel argument values. Used during validation.
  */
using ValueComparator = std::function<bool(const void* /*result*/, const void* /*reference*/)>;

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

/** Kernel definition id returned by kernel definition addition methods in case of an error.
  */
inline const KernelDefinitionId InvalidKernelDefinitionId = std::numeric_limits<KernelDefinitionId>::max();

/** Kernel id returned by kernel addition methods in case of an error.
  */
inline const KernelId InvalidKernelId = std::numeric_limits<KernelId>::max();

/** Argument id returned by argument addition methods in case of an error.
  */
inline const ArgumentId InvalidArgumentId = std::numeric_limits<ArgumentId>::max();

/** Invalid duration used during initialization and in case of an error.
  */
inline const Nanoseconds InvalidDuration = std::numeric_limits<Nanoseconds>::max();

} // namespace ktt
