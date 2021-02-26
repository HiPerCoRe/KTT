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
    /** Tuner output has JSON format. Both serialization and deserialization is supported.
      */
    JSON,

    /** Tuner output has XML format. Both serialization and deserialization is supported.
      */
    XML,

    /** Tuner output has CSV format. Only serialization is supported.
      */
    CSV
};

} // namespace ktt
