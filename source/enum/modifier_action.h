/** @file modifier_action.h
  * Definition of enum for modifier action for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ModifierAction
  * Enum for modifier action for kernel parameters which modify thread size or local memory size.
  */
enum class ModifierAction
{
    /** Kernel parameter value will be added to corresponding kernel thread size or local memory size.
      */
    Add,

    /** Kernel parameter value will be subtracted from corresponding kernel thread size or local memory size.
      */
    Subtract,

    /** Corresponding kernel thread size or local memory size will be multiplied by kernel parameter value.
      */
    Multiply,

    /** Corresponding kernel thread size or local memory size will be divided by kernel parameter value.
      */
    Divide
};

} // namespace ktt
