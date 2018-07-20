/** @file logging_level.h
  * Definition of enum for verbosity level of internal logger.
  */
#pragma once

namespace ktt
{

/** @enum LoggingLevel
  * Enum for verbosity level of internal logger. Higher logging levels also include logging of information from lower levels.
  */
enum class LoggingLevel
{
    /** Logging is completely turned off.
      */
    Off,

    /** Logs information about major problems which usually lead to application termination.
      */
    Error,

    /** Logs information about minor problems which possibly lead to incorrect application behaviour.
      */
    Warning,

    /** Logs general information about application status.
      */
    Info,

    /** Logs detailed information which is useful for debugging.
      */
    Debug
};

} // namespace ktt
