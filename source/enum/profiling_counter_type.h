/** @file profiling_counter_type.h
  * Definition of enum which specifies data type of a profiling counter.
  */
#pragma once

namespace ktt
{

/** @enum ProfilingCounterType
  * Enum which specifies data type of a profiling counter.
  */
enum class ProfilingCounterType
{
    /** Profiling counter is a signed 64-bit integer.
      */
    Int,

    /** Profiling counter is an unsigned 64-bit integer.
      */
    UnsignedInt,

    /** Profiling counter is a 64-bit float.
      */
    Double,

    /** Profiling counter is a 64-bit float with a range of values between 0.0 and 100.0 (corresponding to 0% - 100%).
      */
    Percent,

    /** Profiling counter is an unsigned 64-bit integer. The unit for throughput value is bytes/second.
      */
    Throughput,

    /** Profiling counter is an unsigned 32-bit integer with a range of values between 0 and 10 (0 corresponds to minimum utilization level,
      * 10 to maximum utilization level).
      */
    UtilizationLevel
};

} // namespace ktt
