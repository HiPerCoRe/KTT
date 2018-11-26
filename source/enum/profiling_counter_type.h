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
    Int,
    UnsignedInt,
    Double,
    Percent,
    Throughput,
    UtilizationLevel
};

} // namespace ktt
