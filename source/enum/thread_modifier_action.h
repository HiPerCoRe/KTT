/** @file thread_modifier_action.h
  * Definition of enum for modifier action for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ThreadModifierAction
  * Enum for modifier action for kernel parameters which modify thread size.
  */
enum class ThreadModifierAction
{
    /** Kernel parameter will add its value to corresponding kernel thread size.
      */
    Add,

    /** Kernel parameter will subtract its value from corresponding kernel thread size.
      */
    Subtract,

    /** Kernel parameter will multiply corresponding kernel thread size by its value.
      */
    Multiply,

    /** Kernel parameter will divide corresponding kernel thread size by its value.
      */
    Divide
};

} // namespace ktt
