/** @file OutputFormat.h
  * Format of tuner output.
  */
#pragma once

namespace ktt
{

/** @enum OutputFormat
  * Enum for format of tuner output.
  */
enum class OutputFormat
{
    /** Tuner output has JSON format.
      */
    JSON,

    /** Tuner output has XML format.
      */
    XML
};

} // namespace ktt
