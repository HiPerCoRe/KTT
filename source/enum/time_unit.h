/** @file time_unit.h
  * @brief Definition of enum for time unit used during printing of kernel results.
  */
#pragma once

namespace ktt
{

/** @enum TimeUnit
  * @brief Enum for time unit used during printing of kernel results.
  */
enum class TimeUnit
{
    /** @brief Timings inside kernel results will be printed in nanoseconds.
      */
    Nanoseconds,

    /** @brief Timings inside kernel results will be printed in microseconds.
      */
    Microseconds,

    /** @brief Timings inside kernel results will be printed in milliseconds.
      */
    Milliseconds,

    /** @brief Timings inside kernel results will be printed in seconds.
      */
    Seconds
};

} // namespace ktt
