/** @file ExceptionReason.h
  * Reason why KTT exception was thrown.
  */
#pragma once

namespace ktt
{

/** @enum ExceptionReason
  * Enum which describes reason why KTT exception was thrown.
  */
enum class ExceptionReason
{
    /** General issue with KTT API usage.
      */
    General,

    /** Kernel source file compilation error.
      */
    CompilerError,

    /** Compute device limits were exceeded (e.g., local size was too large, shared memory usage was too high).
      */
    DeviceLimitsExceeded
};

} // namespace ktt
