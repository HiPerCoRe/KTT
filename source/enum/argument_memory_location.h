/** @file argument_memory_location.h
  * Definition of enum for memory location of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentMemoryLocation
  * Enum for memory location of kernel arguments. Specifies the memory from which the argument data will be accessed by compute API functions
  * and kernels.
  */
enum class ArgumentMemoryLocation
{
    /** Argument data will be accessed from device memory. This is recommended setting for devices with dedicated memory, eg. discrete GPUs.
      */
    Device,

    /** Argument data will be accessed from host memory. This is recommended setting for CPUs and devices without dedicated memory,
      * eg. integrated GPUs.
      */
    Host,

    /** Argument data will be accessed from host memory without explicitly creating additional compute API buffer. This flag cannot be used
      * for writable arguments during regular kernel tuning. It can be used for any arguments during kernel tuning by step and kernel running. Note
      * that even when this flag is used, extra buffer copy is still sometimes created internally by compute API. This behaviour depends on particular
      * API and device.
      */
    HostZeroCopy
};

} // namespace ktt
