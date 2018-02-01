/** @file argument_access_type.h
  * Definition of enum for access type of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentAccessType
  * Enum for access type of kernel arguments. Specifies whether kernel argument is used for input or output by compute API kernel function.
  */
enum class ArgumentAccessType
{
    /** Specifies that kernel argument is read-only. Attempting to modify the argument may result in error.
      */
    ReadOnly,

    /** Specifies that kernel argument is write-only. Attempting to read the argument may result in error.
      */
    WriteOnly,

    /** Specifies that kernel argument can be both read and modified.
      */
    ReadWrite
};

} // namespace ktt
