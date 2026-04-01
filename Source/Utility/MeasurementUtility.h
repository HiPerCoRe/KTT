/** @file MeasurementUtility.h
  * Utility functions for precise kernel measurement with iterative execution.
  */
#pragma once

#include <Api/Configuration/PreciseMeasurementParameters.h>
#include <Utility/Timer/Timer.h>
#include <Utility/Logger/Logger.h>
#include <functional>
#include <vector>
#include <string>

namespace ktt
{

/** @struct MeasurementExecutionResult
  * Result of a measurement execution including timing statistics.
  */
struct MeasurementExecutionResult
{
    /** Duration samples collected during measurement. */
    std::vector<Nanoseconds> durationSamples;
    /** Calculated duration based on the selected method. */
    Nanoseconds duration;
    /** Standard deviation of duration samples. */
    double standardDeviation;
    /** Total number of kernel executions performed. */
    unsigned long long executionCount;
    /** Whether maxTimeMs warning was logged. */
    bool maxTimeWarningLogged;
};

/** @class MeasurementUtility
  * Utility class for executing kernels with precise timing measurement.
  * Provides common functionality for iterative kernel execution across different compute engines.
  */
class MeasurementUtility
{
public:
    /** Function type for kernel execution that returns duration in nanoseconds. */
    using KernelExecutor = std::function<Nanoseconds()>;

    /** Executes a kernel multiple times for stable timing measurement.
      * @param executor Function that executes the kernel and returns duration.
      * @param params Measurement parameters controlling execution.
      * @param backendName Name of the compute backend for logging purposes.
      * @return Result containing duration samples and statistics.
      * 
      * The executor is called repeatedly until at least minTimeMs milliseconds have elapsed.
      * Duration is calculated according to params.durationCalculationMethod.
      * Standard deviation is computed from all samples.
      */
    static MeasurementExecutionResult ExecuteWithStableTiming(
        const KernelExecutor& executor,
        const PreciseMeasurementParameters& params,
        const std::string& backendName);

    /** Logs a warning if maxTimeMs > minTimeMs (parameter is ignored in timing-only mode).
      * @param params Measurement parameters to check.
      * @param backendName Name of the compute backend for the warning message.
      * @return True if warning was logged, false otherwise.
      */
    static bool LogMaxTimeWarningIfNeeded(const PreciseMeasurementParameters& params,
        const std::string& backendName);
};

} // namespace ktt
