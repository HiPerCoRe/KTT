/** @file PreciseMeasurementParameters.h
  * Definition of precise measurement parameters for robust kernel sampling.
  */
#pragma once

#include <cstdint>
#include <vector>
#include <KttPlatform.h>
#include <KttTypes.h>

namespace ktt
{

/** @enum DurationCalculationMethod
  * Enum which specifies how to calculate kernel duration from multiple samples
  * during precise measurement.
  */
enum class KTT_API DurationCalculationMethod
{
    /** Use the minimum duration from all samples. */
    Minimum,
    /** Use the median duration from all samples. */
    Median,
    /** Use the average (mean) duration from all samples. */
    Average
};

/** @struct DurationMeasurementResult
  * Struct which holds the result of duration measurement from multiple samples.
  */
struct KTT_API DurationMeasurementResult
{
    /** Calculated duration based on the selected method. */
    Nanoseconds duration;
    /** Standard deviation of all duration samples. */
    double standardDeviation;
    /** Number of samples collected. */
    size_t sampleCount;
};

/** @struct PreciseMeasurementParameters
  * Struct which holds parameters for precise measurement during kernel execution.
  * When power measurement is enabled (KTT built with --power-usage), the kernel
  * is executed multiple times until power readings stabilize within the specified
  * tolerance. When power measurement is not available, the kernel is still executed
  * multiple times for the specified minimum time to compute a more stable duration
  * according to the selected calculation method.
  */
struct KTT_API PreciseMeasurementParameters
{
    /** Minimum measurement time in milliseconds. Measurement continues until
      * at least this much time has elapsed since the first measurement. */
    uint64_t minTimeMs;

    /** Maximum measurement time in milliseconds. When power measurement is enabled,
      * measurement stops when this time is exceeded, even if stability has not been reached.
      * When power measurement is not available, this parameter is ignored (see maxPowerDiff). */
    uint64_t maxTimeMs;

    /** Maximum tolerated relative difference in power readings (0.0 - 1.0).
      * When all recent samples are within this relative difference from the average,
      * measurement is considered stable. Only used when power measurement is enabled
      * (KTT built with --power-usage). When power measurement is not available,
      * this parameter is ignored. */
    double maxPowerDiff;

    /** Method used to calculate kernel duration from multiple samples.
      * Defaults to Minimum for backward compatibility. */
    DurationCalculationMethod durationCalculationMethod;

    /** Default constructor with conservative default values.
      * Defaults: minTime=2000ms, maxTime=20000ms (10x minTime), maxPowerDiff=0.005 (0.5%),
      * durationCalculationMethod=Minimum */
    PreciseMeasurementParameters();

    /** Constructor with custom values.
      * @param minTimeMs Minimum measurement time in milliseconds.
      * @param maxTimeMs Maximum measurement time in milliseconds.
      * @param maxPowerDiff Maximum tolerated relative power difference (0.0 - 1.0).
      * @param durationCalculationMethod Method to calculate duration from samples (default: Minimum). */
    PreciseMeasurementParameters(uint64_t minTimeMs, uint64_t maxTimeMs, double maxPowerDiff,
        DurationCalculationMethod durationCalculationMethod = DurationCalculationMethod::Minimum);

    /** Checks if the parameters are valid.
      * @return True if parameters are valid (minTime > 0, maxTime >= minTime, 0 < maxPowerDiff <= 1.0). */
    bool IsValid() const;

    /** Calculates kernel duration from multiple samples based on the selected method.
      * @param samples Vector of duration samples in nanoseconds.
      * @param method Method to use for duration calculation.
      * @return Calculated duration in nanoseconds. */
    static Nanoseconds CalculateDuration(const std::vector<Nanoseconds>& samples,
        DurationCalculationMethod method);

    /** Calculates standard deviation of duration samples.
      * @param samples Vector of duration samples in nanoseconds.
      * @return Standard deviation of the samples. */
    static double CalculateStandardDeviation(const std::vector<Nanoseconds>& samples);

    /** Calculates both duration and standard deviation from samples.
      * @param samples Vector of duration samples in nanoseconds.
      * @param method Method to use for duration calculation.
      * @return Result containing duration, standard deviation, and sample count. */
    static DurationMeasurementResult ComputeDurationAndStdev(const std::vector<Nanoseconds>& samples,
        DurationCalculationMethod method);
};

} // namespace ktt
