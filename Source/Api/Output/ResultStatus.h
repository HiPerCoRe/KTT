/** @file ResultStatus.h
  * Status of a kernel result.
  */
#pragma once

namespace ktt
{

/** @enum ResultStatus
  * Enum which describes status of a kernel result.
  */
enum class ResultStatus
{
    /** Computation was completed successfully.
      */
    Ok,

    /** Computation failed (e.g., due to invalid source code or some other compute API error).
      */
    ComputationFailed,

    /** Computation was completed successfully, but its output does not match the expected output.
      */
    ValidationFailed
};

} // namespace ktt
