/** @file ComputeApi.h
  * Compute APIs supported by KTT framework.
  */
#pragma once

namespace ktt
{

/** @enum ComputeApi
  * Enum for compute API used by KTT framework. It is utilized during tuner creation.
  */
enum class ComputeApi
{
    /** Tuner will use OpenCL as compute API.
      */
    OpenCL = 1,

    /** Tuner will use CUDA as compute API.
      */
    CUDA = 2,

    /** Tuner will use Vulkan as compute API.
    */
    Vulkan = 3,

    /** Tuner will use C++ as compute API (CPU execution).
    */
    Cpp = 4
};

} // namespace ktt
