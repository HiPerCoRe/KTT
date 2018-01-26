/** @file compute_api.h
  * Definition of enum for compute APIs supported by KTT library.
  */
#pragma once

namespace ktt
{

/** @enum ComputeApi
  * Enum for compute API used by KTT library. It is utilized during tuner creation.
  */
enum class ComputeApi
{
    /** Tuner will use OpenCL as compute API.
      */
    Opencl,

    /** Tuner will use CUDA as compute API.
      */
    Cuda
};

} // namespace ktt
