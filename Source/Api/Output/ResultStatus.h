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

    /** Computation failed due to generic compute API error.
      */
    ComputationFailed,

    /** Computation was completed successfully, but its output does not match the expected output.
      */
    ValidationFailed,

    /** Kernel source file failed to compile.
      */
    CompilationFailed,

    /** Computation could not launch due to device limits being exceeded (e.g., local size was too large).
      */
    DeviceLimitsExceeded
};

} // namespace ktt
