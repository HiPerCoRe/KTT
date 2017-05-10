#include "result_printer.h"

namespace ktt
{

ResultPrinter::ResultPrinter() :
    timeUnit(TimeUnit::Microseconds)
{}

void ResultPrinter::printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const
{
    if (resultMap.find(kernelId) == resultMap.end())
    {
        throw std::runtime_error(std::string("No tuning results found for kernel with id: ") + std::to_string(kernelId));
    }

    auto results = resultMap.find(kernelId)->second;
    if (results.size() == 0)
    {
        throw std::runtime_error(std::string("No tuning results found for kernel with id: ") + std::to_string(kernelId));
    }

    switch (printFormat)
    {
    case PrintFormat::CSV:
        printCSV(results, outputTarget);
        break;
    default:
        printVerbose(results, outputTarget);
    }
}

void ResultPrinter::setResult(const size_t kernelId, const std::vector<TuningResult>& result)
{
    if (resultMap.find(kernelId) != resultMap.end())
    {
        resultMap.erase(kernelId);
    }
    resultMap.insert(std::make_pair(kernelId, result));
}

void ResultPrinter::setTimeUnit(const TimeUnit& timeUnit)
{
    this->timeUnit = timeUnit;
}

void ResultPrinter::printVerbose(const std::vector<TuningResult>& results, std::ostream& outputTarget) const
{
    for (const auto& result : results)
    {
        outputTarget << "Result for kernel <" << result.getKernelName() << ">, configuration: " << std::endl << result.getConfiguration();
        outputTarget << "Kernel duration: " << convertTime(result.getKernelDuration(), timeUnit) << getTimeUnitTag(timeUnit) << std::endl;
        if (result.getManipulatorDuration() != 0)
        {
            outputTarget << "Total duration: " << convertTime(result.getTotalDuration(), timeUnit) << getTimeUnitTag(timeUnit) << std::endl;
        }
        outputTarget << std::endl;
    }

    auto bestResult = getBestResult(results);
    outputTarget << "Best result: " << std::endl;
    outputTarget << "Configuration: " << bestResult.getConfiguration();
    outputTarget << "Kernel duration: " << convertTime(bestResult.getKernelDuration(), timeUnit) << getTimeUnitTag(timeUnit) << std::endl;
    if (bestResult.getManipulatorDuration() != 0)
    {
        outputTarget << "Total duration: " << convertTime(bestResult.getTotalDuration(), timeUnit) << getTimeUnitTag(timeUnit) << std::endl;
    }
    outputTarget << std::endl;
}

void ResultPrinter::printCSV(const std::vector<TuningResult>& results, std::ostream& outputTarget) const
{
    // Header
    outputTarget << "Kernel name;";
    if (results.at(0).getManipulatorDuration() != 0)
    {
        outputTarget << "Total duration (" << getTimeUnitTag(timeUnit) << ");";
    }
    outputTarget << "Kernel duration (" << getTimeUnitTag(timeUnit) << ");Global size;Local size;Threads;";

    auto parameters = results.at(0).getConfiguration().getParameterValues();
    for (const auto& parameter : parameters)
    {
        outputTarget << std::get<0>(parameter) << ";";
    }
    outputTarget << std::endl;

    // Values
    for (const auto& result : results)
    {
        auto configuration = result.getConfiguration();
        auto global = configuration.getGlobalSize();
        auto local = configuration.getLocalSize();

        outputTarget << result.getKernelName() << ";";
        if (results.at(0).getManipulatorDuration() != 0)
        {
            outputTarget << convertTime(result.getTotalDuration(), timeUnit) << ";";
        }
        outputTarget << convertTime(result.getKernelDuration(), timeUnit) << ";";
        outputTarget << std::get<0>(global) << " " << std::get<1>(global) << " " << std::get<2>(global) << ";";
        outputTarget << std::get<0>(local) << " " << std::get<1>(local) << " " << std::get<2>(local) << ";";
        outputTarget << std::get<0>(local) * std::get<1>(local) * std::get<2>(local) << ";";

        auto parameterValues = configuration.getParameterValues();
        for (const auto& value : parameterValues)
        {
            outputTarget << std::get<1>(value) << ";";
        }
        outputTarget << std::endl;
    }
}

TuningResult ResultPrinter::getBestResult(const std::vector<TuningResult>& results) const
{
    TuningResult bestResult = results.at(0);

    for (const auto& result : results)
    {
        if (result.getTotalDuration() < bestResult.getTotalDuration())
        {
            bestResult = result;
        }
    }

    return bestResult;
}

uint64_t ResultPrinter::convertTime(const uint64_t timeInNanoseconds, const TimeUnit& targetUnit) const
{
    switch (targetUnit)
    {
    case TimeUnit::Nanoseconds:
        return timeInNanoseconds;
    case TimeUnit::Microseconds:
        return timeInNanoseconds / 1'000;
    case TimeUnit::Milliseconds:
        return timeInNanoseconds / 1'000'000;
    case TimeUnit::Seconds:
        return timeInNanoseconds / 1'000'000'000;
    default:
        throw std::runtime_error("Unknown time unit");
    }
}

std::string ResultPrinter::getTimeUnitTag(const TimeUnit& timeUnit) const
{
    switch (timeUnit)
    {
    case TimeUnit::Nanoseconds:
        return std::string("ns");
    case TimeUnit::Microseconds:
        return std::string("us");
    case TimeUnit::Milliseconds:
        return std::string("ms");
    case TimeUnit::Seconds:
        return std::string("s");
    default:
        throw std::runtime_error("Unknown time unit");
    }
}

} // namespace ktt
