/** @file global_size_type.h
  * @brief Definition of enum for format of global thread size.
  */
#pragma once

namespace ktt
{

/** @enum GlobalSizeType
  * @brief Enum for format of global thread size. Specifies the format of global thread size specified by user during kernel addition.
  */
enum class GlobalSizeType
{
    /** @brief Global thread size uses OpenCL format for NDRange dimensions specification.
      */
    Opencl,

    /** @brief Global thread size uses CUDA format for grid dimensions specification.
      */
    Cuda
};

} // namespace ktt
