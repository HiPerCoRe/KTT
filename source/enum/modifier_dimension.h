/** @file modifier_dimension.h
  * Definition of enum for modifier dimension for kernel parameters.
  */
#pragma once

namespace ktt
{

/** @enum ModifierDimension
  * Enum for modifier dimension for kernel parameters. Dimensions are utilized during specification of parameters which modify kernel thread sizes.
  */
enum class ModifierDimension
{
    /** Kernel parameter will modify thread size in dimension X.
      */
    X,

    /** Kernel parameter will modify thread size in dimension Y.
      */
    Y,

    /** Kernel parameter will modify thread size in dimension Z.
      */
    Z
};

} // namespace ktt
