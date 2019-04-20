/** @file validation_mode.h
  * Definition of enum for enabling kernel output validation in different scenarios.
  */
#pragma once

#include <enum/enum_bitfield.h>

namespace ktt
{

/** @enum ValidationMode
  * Enum for enabling kernel output validation in different scenarios.
  */
enum class ValidationMode
{
    /** Kernel output validation is completely disabled.
      */
    None = 0,

    /** Kernel output is validated during kernel running.
      */
    Running = (1 << 0),

    /** Kernel output is validated during offline kernel tuning.
      */
    OfflineTuning = (1 << 1),

    /** Kernel output is validated during online kernel tuning.
      */
    OnlineTuning = (1 << 2),

    /** Kernel output is always validated.
      */
    All = Running | OfflineTuning | OnlineTuning
};

/** Validation mode enum supports bitwise operations.
  */
template <>
struct EnableBitfieldOperators<ValidationMode>
{
    /** Bitwise operations are enabled.
      */
    static const bool enable = true;
};

} // namespace ktt
