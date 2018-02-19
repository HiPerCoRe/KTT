/** @file argument_data_type.h
  * Definition of enum for data type of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentDataType
  * Enum for data type of kernel arguments. Specifies the data type of elements inside single kernel argument.
  */
enum class ArgumentDataType
{
    /** 8-bit signed integer type.
      */
    Char,

    /** 8-bit unsigned integer type.
      */
    UnsignedChar,

    /** 16-bit signed integer type.
      */
    Short,

    /** 16-bit unsigned integer type.
      */
    UnsignedShort,

    /** 32-bit signed integer type.
      */
    Int,

    /** 32-bit unsigned integer type.
      */
    UnsignedInt,

    /** 64-bit signed integer type.
      */
    Long,

    /** 64-bit unsigned integer type.
      */
    UnsignedLong,

    /** 16-bit floating-point type.
      */
    Half,

    /** 32-bit floating-point type.
      */
    Float,

    /** 64-bit floating-point type.
      */
    Double,

    /** Custom data type, usually defined by user. Custom data type has to be trivially copyable. It can be for example struct or class.
      */
    Custom
};

} // namespace ktt
