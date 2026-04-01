#include <Utility/MeasurementUtility.h>

namespace ktt
{

MeasurementExecutionResult MeasurementUtility::ExecuteWithStableTiming(
    const KernelExecutor& executor,
    const PreciseMeasurementParameters& params,
    const std::string& backendName)
{
    MeasurementExecutionResult result;
    result.durationSamples.reserve(100); // Pre-allocate for efficiency
    result.maxTimeWarningLogged = false;

    // Execute kernel at least once to get initial sample
    result.durationSamples.push_back(executor());
    result.executionCount = 1;

    Timer execTimer;
    execTimer.Start();

    // Log warning if maxTimeMs > minTimeMs (maxTimeMs is ignored in timing-only mode)
    if (params.maxTimeMs > params.minTimeMs)
    {
        Logger::LogWarning("Power measurement is not available for " + backendName + " backend. "
                           "maxTimeMs parameter is ignored; only minTimeMs is used for stable timing measurement.");
        result.maxTimeWarningLogged = true;
    }

    // Run kernel multiple times until at least minTimeMs has elapsed
    while (execTimer.GetCheckpointTime() < static_cast<unsigned long>(params.minTimeMs) * 1000000ULL)
    {
        result.durationSamples.push_back(executor());
        result.executionCount++;
    }

    // Calculate duration and standard deviation using the utility methods
    const auto measurementResult = PreciseMeasurementParameters::ComputeDurationAndStdev(
        result.durationSamples, params.durationCalculationMethod);
    
    result.duration = measurementResult.duration;
    result.standardDeviation = measurementResult.standardDeviation;

    Logger::LogInfo("Stable timing measured from " + std::to_string(result.durationSamples.size()) + 
                    " kernel runs (out of " + std::to_string(result.executionCount) + " total executions)");

    return result;
}

bool MeasurementUtility::LogMaxTimeWarningIfNeeded(const PreciseMeasurementParameters& params,
    const std::string& backendName)
{
    if (params.maxTimeMs > params.minTimeMs)
    {
        Logger::LogWarning("Power measurement is not available for " + backendName + " backend. "
                           "maxTimeMs parameter is ignored; only minTimeMs is used for stable timing measurement.");
        return true;
    }
    return false;
}

} // namespace ktt
