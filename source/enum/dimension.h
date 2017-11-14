/** @file dimension.h
  * @brief Definition of enum for dimension.
  */
#pragma once

namespace ktt
{

/** @enum Dimension
  * @brief Enum for dimensions. Dimensions are utilized during specification of parameters which modify kernel thread sizes.
  */
enum class Dimension
{
    /**  @brief Kernel parameter will modify thread size in dimension X.
      */
    X,

    /**  @brief Kernel parameter will modify thread size in dimension Y.
      */
    Y,

    /**  @brief Kernel parameter will modify thread size in dimension Z.
      */
    Z
};

} // namespace ktt
