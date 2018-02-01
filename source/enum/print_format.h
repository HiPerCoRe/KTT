/** @file print_format.h
  * Definition of enum for format of printed results.
  */
#pragma once

namespace ktt
{

/** @enum PrintFormat
  * Enum for format of printed results. Specifies the format used during printing of tuning results.
  */
enum class PrintFormat
{
    /** Format suitable for printing to console or log file.
      */
    Verbose,

    /** Format suitable for printing into CSV file, allows easier data analysis and vizualization.
      */
    CSV
};

} // namespace ktt
