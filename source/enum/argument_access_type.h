/** @file argument_access_type.h
  * @brief Definition of enum for access type of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentAccessType
  * @brief Enum for access type of kernel arguments. Specifies whether kernel argument is used for input or output by compute API kernel function.
  */
enum class ArgumentAccessType
{
    /** @brief Specifies that kernel argument is read-only. Attempting to modify the argument may result in error.
      */
    ReadOnly,

    /** @brief Specifies that kernel argument is write-only. Attempting to read the argument may result in error.
      */
    WriteOnly,

    /** @brief Specifies that kernel argument can be both read and modified.
      */
    ReadWrite
};

} // namespace ktt
