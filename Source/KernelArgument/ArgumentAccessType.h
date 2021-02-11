/** @file ArgumentAccessType.h
  * Access type of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentAccessType
  * Enum for access type of kernel arguments. Specifies whether argument is used for input or output by compute API
  * kernel functions.
  */
enum class ArgumentAccessType
{
    /** Kernel argument access is undefined.
      */
    Undefined,

    /** Kernel argument is used for input.
      */
    ReadOnly,

    /** Kernel argument is used for output.
      */
    WriteOnly,

    /** Kernel argument is used for both input and output.
      */
    ReadWrite
};

} // namespace ktt
