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
    /** @brief Calculates sum of differences between each pair of elements, then compares the sum to specified threshold.
      */
    AbsoluteDifference,

    /** @brief Calculates difference for each pair of elements, then compares the difference to specified threshold.
      */
    SideBySideComparison,

    /** @brief Calculates difference for each pair of elements, then compares the difference divided by reference value to specified threshold.
      */
    SideBySideRelativeComparison
};

} // namespace ktt
