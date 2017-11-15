/** @file thread_modifier_action.h
  * @brief Definition of enum for modifier action for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ThreadModifierAction
  * @brief Enum for modifier action for kernel parameters which modify thread size.
  */
enum class ThreadModifierAction
{
    /** @brief Kernel parameter will add its value to corresponding kernel thread size.
      */
    Add,

    /** @brief Kernel parameter will subtract its value from corresponding kernel thread size.
      */
    Subtract,

    /** @brief Kernel parameter will multiply corresponding kernel thread size by its value.
      */
    Multiply,

    /** @brief Kernel parameter will divide corresponding kernel thread size by its value.
      */
    Divide
};

} // namespace ktt
