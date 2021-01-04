/** @file dimension_vector_type.h
  * Definition of enum for dimension vector type.
  */
#pragma once

namespace ktt
{

/** @enum DimensionVectorType
  * Enum for dimension vector type. Specifies whether a single dimension vector holds global or local kernel thread dimensions.
  */
enum class DimensionVectorType
{
    /** Dimension vector holds global kernel thread dimensions.
      */
    Global,

    /** Dimension vector holds local kernel thread dimensions.
      */
    Local
};

} // namespace ktt
