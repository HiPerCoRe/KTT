/** @file KernelCompilationData.h
  * Compilation information about specific kernel configuration.
  */
#pragma once

#include <cstdint>

#include <KttPlatform.h>

namespace ktt
{

/** @struct KernelCompilationData
  * Structure which holds compilation information about specific kernel configuration.
  */
struct KTT_API KernelCompilationData
{
public:
    /** @fn KernelCompilationData()
      * Constructor which initializes all data values to zero.
      */
    KernelCompilationData();

    /** The maximum size of a work-group (thread block in CUDA), beyond which a launch of the kernel would fail. Depends on both
      * kernel and device.
      */
    uint64_t m_MaxWorkGroupSize;

    /** The size in bytes of statically-allocated local memory (shared memory in CUDA) required by the kernel.
      */
    uint64_t m_LocalMemorySize;

    /**
      * The size in bytes of private memory (local memory in CUDA) used by each work-item of the kernel.
      */
    uint64_t m_PrivateMemorySize;

    /**
      * The size in bytes of user-allocated constant memory required by the kernel. This value is valid only for CUDA backend.
      */
    uint64_t m_ConstantMemorySize;

    /**
      * The number of registers used by each work-item of the kernel. This value is valid only for CUDA backend.
      */
    uint64_t m_RegistersCount;
};

} // namespace ktt
