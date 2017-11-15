/** @file thread_modifier_type.h
  * @brief Definition of enum for modifier type for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ThreadModifierType
  * @brief Enum for modifier type for kernel parameters. Specifies whether kernel parameter value affects corresponding kernel thread size.
  */
enum class ThreadModifierType
{
    /** @brief Parameter value does not affect any thread sizes of corresponding kernel.
      */
    None,

    /** @brief Parameter value affects global thread size of corresponding kernel.
      */
    Global,

    /** @brief Parameter value affects local thread size of corresponding kernel.
      */
    Local
};

} // namespace ktt
