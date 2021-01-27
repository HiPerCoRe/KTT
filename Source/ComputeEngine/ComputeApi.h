/** @file ComputeApi.h
  * Compute APIs supported by KTT framework.
  */
#pragma once

namespace ktt
{

/** @enum ComputeAPI
  * Enum for compute APIs used by KTT framework. It is utilized during tuner creation.
  */
enum class ComputeAPI
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
