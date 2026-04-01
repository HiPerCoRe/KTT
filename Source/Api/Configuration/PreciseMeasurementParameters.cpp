#include <Api/Configuration/PreciseMeasurementParameters.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ktt
{

PreciseMeasurementParameters::PreciseMeasurementParameters() :
    minTimeMs(2000),
    maxTimeMs(20000),
    maxPowerDiff(0.005),
    durationCalculationMethod(DurationCalculationMethod::Minimum)
{}

PreciseMeasurementParameters::PreciseMeasurementParameters(uint64_t minTimeMs, uint64_t maxTimeMs, double maxPowerDiff,
    DurationCalculationMethod durationCalculationMethod) :
    minTimeMs(minTimeMs),
    maxTimeMs(maxTimeMs),
    maxPowerDiff(maxPowerDiff),
    durationCalculationMethod(durationCalculationMethod)
{}

bool PreciseMeasurementParameters::IsValid() const
{
    return minTimeMs > 0 && maxTimeMs >= minTimeMs && maxPowerDiff > 0.0 && maxPowerDiff <= 1.0;
}

Nanoseconds PreciseMeasurementParameters::CalculateDuration(const std::vector<Nanoseconds>& samples,
    DurationCalculationMethod method)
{
    if (samples.empty())
    {
        return 0;
    }

    switch (method)
    {
        case DurationCalculationMethod::Minimum:
            return *std::min_element(samples.begin(), samples.end());
        case DurationCalculationMethod::Median:
        {
            std::vector<Nanoseconds> sortedSamples = samples;
            std::sort(sortedSamples.begin(), sortedSamples.end());
            if (sortedSamples.size() % 2 == 0)
            {
                return (sortedSamples[sortedSamples.size() / 2 - 1] + sortedSamples[sortedSamples.size() / 2]) / 2;
            }
            else
            {
                return sortedSamples[sortedSamples.size() / 2];
            }
        }
        case DurationCalculationMethod::Average:
            return std::accumulate(samples.begin(), samples.end(), 0ULL) / samples.size();
        default:
            return *std::min_element(samples.begin(), samples.end());
    }
}

double PreciseMeasurementParameters::CalculateStandardDeviation(const std::vector<Nanoseconds>& samples)
{
    if (samples.empty())
    {
        return 0.0;
    }

    const double mean = static_cast<double>(std::accumulate(samples.begin(), samples.end(), 0ULL)) / samples.size();
    double varianceSum = 0.0;
    for (const auto& sample : samples)
    {
        const double diff = static_cast<double>(sample) - mean;
        varianceSum += diff * diff;
    }
    return std::sqrt(varianceSum / samples.size());
}

DurationMeasurementResult PreciseMeasurementParameters::ComputeDurationAndStdev(const std::vector<Nanoseconds>& samples,
    DurationCalculationMethod method)
{
    DurationMeasurementResult result;
    result.duration = CalculateDuration(samples, method);
    result.standardDeviation = CalculateStandardDeviation(samples);
    result.sampleCount = samples.size();
    return result;
}

} // namespace ktt
