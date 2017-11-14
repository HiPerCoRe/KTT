/** @file argument_data_type.h
  * @brief Definition of enum for data type of kernel arguments.
  */
#pragma once

namespace ktt
{

/** @enum ArgumentDataType
  * @brief Enum for data type of kernel arguments. Specifies the data type of elements inside single kernel argument.
  */
enum class ArgumentDataType
{
    /** @brief 8-bit signed integer type.
      */
    Char,

    /** @brief 8-bit unsigned integer type.
      */
    UnsignedChar,

    /** @brief 16-bit signed integer type.
      */
    Short,

    /** @brief 16-bit unsigned integer type.
      */
    UnsignedShort,

    /** @brief 32-bit signed integer type.
      */
    Int,

    /** @brief 32-bit unsigned integer type.
      */
    UnsignedInt,

    /** @brief 64-bit signed integer type.
      */
    Long,

    /** @brief 64-bit unsigned integer type.
      */
    UnsignedLong,

    /** @brief 16-bit floating-point type.
      */
    Half,

    /** @brief 32-bit floating-point type.
      */
    Float,

    /** @brief 64-bit floating-point type.
      */
    Double,

    /** @brief Custom data type, usually defined by user. Custom data type has to be trivially copyable. It can be for example struct or class.
      */
    Custom
};

} // namespace ktt
