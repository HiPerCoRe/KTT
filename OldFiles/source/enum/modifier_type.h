/** @file modifier_type.h
  * Definition of enum for modifier type for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ModifierType
  * Enum for modifier type for kernel parameters. Specifies whether kernel parameter value affects corresponding kernel thread size.
  */
enum class ModifierType
{
    /** Parameter value affects global thread size of corresponding kernel.
      */
    Global,

    /** Parameter value affects local thread size of corresponding kernel.
      */
    Local
};

} // namespace ktt
