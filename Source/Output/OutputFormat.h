/** @file OutputFormat.h
  * Format of kernel result output.
  */
#pragma once

namespace ktt
{

/** @enum OutputFormat
  * Enum for format of kernel result output.
  */
enum class OutputFormat
{
    /** Results are printed in JSON format.
      */
    JSON,

    /** Results are printed in XML format.
      */
    XML,

    /** Results are printed in CSV format.
      */
    CSV
};

} // namespace ktt
