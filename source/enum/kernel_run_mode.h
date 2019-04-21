/** @file kernel_run_mode.h
  * Definition of enum for differentiating between different kernel running scenarios.
  */
#pragma once

namespace ktt
{

/** @enum KernelRunMode
  * Enum for differentiating between different kernel running scenarios.
  */
enum class KernelRunMode
{
    /** Regular kernel running.
      */
    Running,

    /** Offline kernel tuning.
      */
    OfflineTuning,

    /** Online kernel tuning.
      */
    OnlineTuning,

    /** Computation of reference output for result validation.
      */
    ResultValidation
};

} // namespace ktt
