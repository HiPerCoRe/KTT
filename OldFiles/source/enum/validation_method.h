/** @file validation_method.h
  * Definition of enum for validation method used during validation of floating-point output arguments.
  */
#pragma once

namespace ktt
{

/** @enum ValidationMethod
  * Enum for validation method used during validation of floating-point output arguments.
  */
enum class ValidationMethod
{
    /** Calculates sum of differences between each pair of elements, then compares the sum to specified threshold.
      */
    AbsoluteDifference,

    /** Calculates difference for each pair of elements, then compares the difference to specified threshold.
      */
    SideBySideComparison,

    /** Calculates difference for each pair of elements, then compares the difference divided by reference value to specified threshold.
      */
    SideBySideRelativeComparison
};

} // namespace ktt
