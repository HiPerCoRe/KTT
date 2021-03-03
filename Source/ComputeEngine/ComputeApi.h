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
    OpenCL,

    /** Tuner will use CUDA as compute API.
      */
    CUDA,

    /** Tuner will use Vulkan as compute API.
    */
    Vulkan
};

} // namespace ktt
