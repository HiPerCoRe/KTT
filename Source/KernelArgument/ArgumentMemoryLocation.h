/** @file ArgumentMemoryLocation.h
  * Memory location of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentMemoryLocation
  * Enum for memory location of vector kernel arguments. Specifies the memory from which the argument data will be accessed by
  * compute API functions and kernels.
  */
enum class ArgumentMemoryLocation
{
    /** Argument data will be accessed from device memory. This is recommended setting for devices with dedicated memory,
      * e.g. discrete GPUs.
      */
    Device,

    /** Argument data will be accessed from host memory. This is recommended setting for CPUs and devices without dedicated
      * memory, e.g. integrated GPUs.
      */
    Host,

    /* Argument data will be stored using compute API's unified memory system. It can be directly accessed from both
     * host and device.
     */
    Unified
};

} // namespace ktt
