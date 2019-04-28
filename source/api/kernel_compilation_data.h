/** @file kernel_compilation_data.h
  * ...
  */
#pragma once

#include <cstdint>
#include <ktt_platform.h>

namespace ktt
{

/** @struct KernelCompilationData
  * ...
  */
struct KTT_API KernelCompilationData
{
public:
    /** @fn KernelCompilationData()
      * ...
      */
    KernelCompilationData();

    uint64_t maxWorkGroupSize;
    uint64_t localMemorySize;
    uint64_t privateMemorySize;
    uint64_t constantMemorySize;
    uint64_t registersCount;
};

} // namespace ktt
