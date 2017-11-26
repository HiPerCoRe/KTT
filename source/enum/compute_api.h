/** @file compute_api.h
  * @brief Definition of enum for compute APIs supported by KTT library.
  */
#pragma once

namespace ktt
{

/** @enum ComputeApi
  * @brief Enum for compute API used by KTT library. It is utilized during tuner creation.
  */
enum class ComputeApi
{
    /** @brief Tuner will use OpenCL as compute API.
      */
    Opencl,

    /** @brief Tuner will use CUDA as compute API.
      */
    Cuda
};

} // namespace ktt
