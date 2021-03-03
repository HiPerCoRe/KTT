/** @file ModifierType.h
  * Modifier type for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ModifierType
  * Enum for modifier type for kernel parameters. Specifies whether kernel parameter value affects corresponding kernel thread size.
  */
enum class ModifierType
{
    /** Parameter value affects global kernel thread size.
      */
    Global,

    /** Parameter value affects local kernel thread size.
      */
    Local
};

} // namespace ktt
