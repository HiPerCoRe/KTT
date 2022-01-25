/** @file ArgumentMemoryType.h
  * Memory type of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentMemoryType
  * Enum for memory type of kernel arguments. Specifies which compute API function should be used internally by KTT framework
  * to make the argument accessible to kernel functions.
  */
enum class ArgumentMemoryType
{
    /** Argument is a scalar. Scalar arguments are made visible to kernels as a local copy.
      */
    Scalar,

    /** Argument is a vector. Vector arguments are made visible to kernels through compute API buffers. See
      * ::ArgumentMemoryLocation for more information.
      */
    Vector,

    /** Argument will be located in local memory. Kernel arguments cannot be directly transferred into local memory from
      * host memory. Assigning local memory argument to kernel from KTT API simply means that the compute API will allocate
      * enough local memory to hold number of elements specified by the argument. The memory then needs to be filled with
      * data on kernel side.
      */
    Local,

    /** Argument corresponds to the CUDA symbol which resides in global or constant device memory and matches the argument's name.
      * In Vulkan and OpenCL, symbol arguments are treated in the same way as scalars.
      */
    Symbol
};

} // namespace ktt
