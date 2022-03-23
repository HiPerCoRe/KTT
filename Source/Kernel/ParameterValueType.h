/** @file ParameterValueType.h
  * Value type for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ParameterValueType
  * Enum for value type for kernel parameters.
  */
enum class ParameterValueType
{
    /** Parameter has 64-bit signed integer type.
      */
    Int,

    /** Parameter has 64-bit unsigned integer type.
      */
    UnsignedInt,

    /** Parameter has 64-bit floating-point type.
      */
    Double,

    /** Parameter has boolean type.
      */
    Bool,

    /** Parameter has string type.
      */
    String
};

} // namespace ktt
