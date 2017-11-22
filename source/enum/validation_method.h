/** @file validation_method.h
  * @brief Definition of enum for validation method used during validation of floating-point output arguments.
  */
#pragma once

namespace ktt
{

/** @enum ValidationMethod
  * @brief Enum for validation method used during validation of floating-point output arguments.
  */
enum class ValidationMethod
{
    /** @brief Calculates sum of all differences between individual element comparisons, then compares this sum to specified threshold.
      */
    AbsoluteDifference,

    /** @brief Calculates difference each time when comparing individual elements, then compares this difference to specified threshold.
      */
    SideBySideComparison,

    /** @brief Calculates difference each time when comparing individual elements, then compares this difference to specified threshold.
      */
    SideBySideRelativeComparison,
};

} // namespace ktt
