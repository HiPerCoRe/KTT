/** @file dimension.h
  * Definition of enum for dimension.
  */
#pragma once

namespace ktt
{

/** @enum Dimension
  * Enum for dimensions. Dimensions are utilized during specification of parameters which modify kernel thread sizes.
  */
enum class Dimension
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
