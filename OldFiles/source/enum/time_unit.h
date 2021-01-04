/** @file time_unit.h
  * Definition of enum for time unit used during printing of kernel results.
  */
#pragma once

namespace ktt
{

/** @enum TimeUnit
  * Enum for time unit used during printing of kernel results.
  */
enum class TimeUnit
{
    /** Times inside kernel results will be printed in nanoseconds.
      */
    Nanoseconds,

    /** Times inside kernel results will be printed in microseconds.
      */
    Microseconds,

    /** Times inside kernel results will be printed in milliseconds.
      */
    Milliseconds,

    /** Times inside kernel results will be printed in seconds.
      */
    Seconds
};

} // namespace ktt
