/** @file ArgumentAccessType.h
  * Access type of kernel arguments.
  */
#pragma once

#include <Utility/BitfieldEnum.h>

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
    Undefined = 0,

    /** Kernel argument is used for input.
      */
    Read = (1 << 0),

    /** Kernel argument is used for output.
      */
    Write = (2 << 0),

    /** Kernel argument is used for both input and output.
      */
    ReadWrite = Read | Write
};

/** Argument access type enum supports bitwise operations.
  */
template <>
struct EnableBitfieldOperators<ArgumentAccessType>
{
    /** Bitwise operations are enabled.
      */
    static const bool m_Enable = true;
};

} // namespace ktt
