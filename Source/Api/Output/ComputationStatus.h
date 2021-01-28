/** @file ComputationStatus.h
  * Status of a finished kernel computation.
  */
#pragma once

namespace ktt
{

/** @enum ComputationStatus
  * Enum which describes status of a finished kernel computation.
  */
enum class ComputationStatus
{
    /** Kernel computation was completed successfully.
      */
    Ok,

    /** Kernel computation failed.
      */
    Failed
};

} // namespace ktt
