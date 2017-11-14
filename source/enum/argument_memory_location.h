/** @file argument_memory_location.h
  * @brief Definition of enum for memory location of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentMemoryLocation
  * @brief Enum for memory location of kernel arguments. Specifies the memory from which the argument data will be accessed by compute API functions
  * and kernels.
  */
enum class ArgumentMemoryLocation
{
    /** @brief Argument data will be accessed from device memory. This is recommended setting for devices with dedicated memory, eg. discrete GPUs.
      */
    Device,

    /** @brief Argument data will be accessed from host memory. This is recommended setting for CPUs and devices without dedicated memory,
      * eg. integrated GPUs.
      */
    Host,

    /** @brief Argument data will be accessed from host memory without explicitly creating additional compute API buffer. This flag cannot be used
      * for writable arguments during regular kernel tuning. It can be used for any arguments during kernel tuning by step and kernel running. Note
      * that even when this flag is used, extra buffer copy is still sometimes created internally by compute API. This bevaiour depends on particular
      * API and device.
      */
    HostZeroCopy
};

} // namespace ktt
