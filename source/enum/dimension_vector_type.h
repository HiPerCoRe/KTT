/** @file dimension_vector_type.h
  * @brief Definition of enum for dimension vector type.
  */
#pragma once

namespace ktt
{

/** @enum DimensionVectorType
  * @brief Enum for dimension vector type. Specifies whether a single dimension vector holds global or local kernel thread dimensions.
  */
enum class DimensionVectorType
{
    /** @brief Dimension vector holds global kernel thread dimensions.
      */
    Global,

    /** @brief Dimension vector holds local kernel thread dimensions.
      */
    Local
};

} // namespace ktt
