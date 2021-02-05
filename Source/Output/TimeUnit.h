/** @file TimeUnit.h
  * Time unit used during logging and output operations.
  */
#pragma once

namespace ktt
{

/** @enum TimeUnit
  * Enum for time unit used during logging and output operations.
  */
enum class TimeUnit
{
    /** Durations will be printed in nanoseconds.
      */
    Nanoseconds,

    /** Durations will be printed in microseconds.
      */
    Microseconds,

    /** Durations will be printed in milliseconds.
      */
    Milliseconds,

    /** Durations will be printed in seconds.
      */
    Seconds
};

} // namespace ktt
