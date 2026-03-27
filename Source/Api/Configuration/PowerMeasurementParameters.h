/** @file PowerMeasurementParameters.h
  * Definition of power measurement parameters for robust power sampling.
  */
#pragma once

#include <cstdint>
#include <KttPlatform.h>

namespace ktt
{

/** @enum DurationCalculationMethod
  * Enum which specifies how to calculate kernel duration from multiple samples
  * during robust power measurement.
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

/** @struct PowerMeasurementParameters
  * Struct which holds parameters for robust power measurement during kernel execution.
  * When power measurement is enabled, the kernel is executed multiple times until
  * power readings stabilize within the specified tolerance.
  */
struct KTT_API PowerMeasurementParameters
{
    /** Minimum measurement time in milliseconds. Power measurement continues until
      * at least this much time has elapsed since the first measurement. */
    uint64_t minTimeMs;

    /** Maximum measurement time in milliseconds. Power measurement stops when this
      * time is exceeded, even if stability has not been reached. */
    uint64_t maxTimeMs;

    /** Maximum tolerated relative difference in power readings (0.0 - 1.0).
      * When all recent samples are within this relative difference from the average,
      * measurement is considered stable. */
    double maxPowerDiff;

    /** Method used to calculate kernel duration from multiple samples.
      * Defaults to Minimum for backward compatibility. */
    DurationCalculationMethod durationCalculationMethod;

    /** Default constructor with conservative default values.
      * Defaults: minTime=2000ms, maxTime=20000ms (10x minTime), maxPowerDiff=0.005 (0.5%),
      * durationCalculationMethod=Minimum */
    PowerMeasurementParameters();

    /** Constructor with custom values.
      * @param minTimeMs Minimum measurement time in milliseconds.
      * @param maxTimeMs Maximum measurement time in milliseconds.
      * @param maxPowerDiff Maximum tolerated relative power difference (0.0 - 1.0).
      * @param durationCalculationMethod Method to calculate duration from samples (default: Minimum). */
    PowerMeasurementParameters(uint64_t minTimeMs, uint64_t maxTimeMs, double maxPowerDiff,
        DurationCalculationMethod durationCalculationMethod = DurationCalculationMethod::Minimum);

    /** Checks if the parameters are valid.
      * @return True if parameters are valid (minTime > 0, maxTime >= minTime, 0 < maxPowerDiff <= 1.0). */
    bool IsValid() const;
};

} // namespace ktt
