/** @file ModifierAction.h
  * Modifier action for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ModifierAction
  * Enum for modifier action for kernel parameters which modify thread size.
  */
enum class ModifierAction
{
    /** Kernel parameter value will be added to corresponding kernel thread size.
      */
    Add,

    /** Kernel parameter value will be subtracted from corresponding kernel thread size.
      */
    Subtract,

    /** Corresponding kernel thread size will be multiplied by kernel parameter value.
      */
    Multiply,

    /** Corresponding kernel thread size will be divided by kernel parameter value.
      */
    Divide,

    /** Corresponding kernel thread size will be divided by kernel parameter value and then rounded up to its multiple.
      */
    DivideCeil
};

} // namespace ktt
