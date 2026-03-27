#include <Api/Configuration/PowerMeasurementParameters.h>

namespace ktt
{

PowerMeasurementParameters::PowerMeasurementParameters() :
    minTimeMs(2000),
    maxTimeMs(20000),
    maxPowerDiff(0.005),
    durationCalculationMethod(DurationCalculationMethod::Minimum)
{}

PowerMeasurementParameters::PowerMeasurementParameters(uint64_t minTimeMs, uint64_t maxTimeMs, double maxPowerDiff,
    DurationCalculationMethod durationCalculationMethod) :
    minTimeMs(minTimeMs),
    maxTimeMs(maxTimeMs),
    maxPowerDiff(maxPowerDiff),
    durationCalculationMethod(durationCalculationMethod)
{}

bool PowerMeasurementParameters::IsValid() const
{
    return minTimeMs > 0 && maxTimeMs >= minTimeMs && maxPowerDiff > 0.0 && maxPowerDiff <= 1.0;
}

} // namespace ktt
