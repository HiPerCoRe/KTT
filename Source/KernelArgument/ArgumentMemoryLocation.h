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
    /** Default memory location for non-vector kernel arguments.
      */
    Undefined,

    /** Argument data will be accessed from device memory. This is recommended setting for devices with dedicated memory,
      * e.g., discrete GPUs.
      */
    Device,

    /** Argument data will be accessed from host memory. This is recommended setting for CPUs and devices without dedicated
      * memory, e.g., integrated GPUs.
      */
    Host,

    /** Argument data will be accessed from host memory without explicitly creating additional compute API buffer. This flag
      * cannot be used for writable arguments during regular kernel tuning. It can be used for any arguments during step kernel tuning
      * and kernel running. Note that even when this flag is used, extra buffer copy is still sometimes created internally by compute API.
      * This behaviour depends on particular API and device.
      */
    HostZeroCopy,

    /* Argument data will be stored using compute API's unified memory system. It can be directly accessed from both
     * host and device.
     */
    Unified
};

} // namespace ktt
