#include <algorithm>

#include "result_printer.h"

namespace ktt
{

ResultPrinter::ResultPrinter() :
    timeUnit(TimeUnit::Microseconds),
    globalSizeType(GlobalSizeType::Opencl),
    printInvalidResult(false)
{}

void ResultPrinter::printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const
{
    if (resultMap.find(kernelId) == resultMap.end())
    {
        throw std::runtime_error(std::string("No tuning results found for kernel with id: ") + std::to_string(kernelId));
    }

    auto results = resultMap.find(kernelId)->second;

    switch (printFormat)
    {
    case PrintFormat::CSV:
        printCsv(results, outputTarget);
        break;
    case PrintFormat::Verbose:
        printVerbose(results, outputTarget);
        break;
    default:
        throw std::runtime_error("Unknown print format");
    }
}

void ResultPrinter::setResult(const size_t kernelId, const std::vector<TuningResult>& results)
{
    if (resultMap.find(kernelId) != resultMap.end())
    {
        resultMap.erase(kernelId);
    }
    resultMap.insert(std::make_pair(kernelId, results));
}

void ResultPrinter::setTimeUnit(const TimeUnit& timeUnit)
{
    this->timeUnit = timeUnit;
}

void ResultPrinter::setGlobalSizeType(const GlobalSizeType& globalSizeType)
{
    this->globalSizeType = globalSizeType;
}

void ResultPrinter::setInvalidResultPrinting(const bool flag)
{
    printInvalidResult = flag;
}

void ResultPrinter::printVerbose(const std::vector<TuningResult>& results, std::ostream& outputTarget) const
{
    for (const auto& result : results)
    {
        if (!result.isValid())
        {
            continue;
        }

        outputTarget << "Result for kernel <" << result.getKernelName() << ">, configuration: " << std::endl;
        printConfigurationVerbose(outputTarget, result.getConfiguration());
        outputTarget << "Kernel duration: " << convertTime(result.getKernelDuration(), timeUnit) << getTimeUnitTag(timeUnit) << std::endl;
        if (result.getManipulatorDuration() != 0)
        {
            outputTarget << "Total duration: " << convertTime(result.getTotalDuration(), timeUnit) << getTimeUnitTag(timeUnit) << std::endl;
        }
        outputTarget << std::endl;
    }

    auto bestResult = getBestResult(results);
    if (bestResult.isValid())
    {
        outputTarget << "Best result: " << std::endl;
        outputTarget << "Configuration: ";
        printConfigurationVerbose(outputTarget, bestResult.getConfiguration());
        outputTarget << "Kernel duration: " << convertTime(bestResult.getKernelDuration(), timeUnit) << getTimeUnitTag(timeUnit) << std::endl;
        if (bestResult.getManipulatorDuration() != 0)
        {
            outputTarget << "Total duration: " << convertTime(bestResult.getTotalDuration(), timeUnit) << getTimeUnitTag(timeUnit) << std::endl;
        }
        outputTarget << std::endl;
    }
    else
    {
        outputTarget << "No best result found" << std::endl;
    }

    if (printInvalidResult)
    {
        for (const auto& result : results)
        {
            if (result.isValid())
            {
                continue;
            }

            outputTarget << "Invalid result for kernel <" << result.getKernelName() << ">, configuration: " << std::endl;
            printConfigurationVerbose(outputTarget, result.getConfiguration());
            outputTarget << "Result status: " << result.getStatusMessage();
            outputTarget << std::endl << std::endl;
        }
    }
}

void ResultPrinter::printCsv(const std::vector<TuningResult>& results, std::ostream& outputTarget) const
{
    // Header
    outputTarget << "Kernel name,";
    if (results.at(0).getManipulatorDuration() != 0)
    {
        outputTarget << "Total duration (" << getTimeUnitTag(timeUnit) << "),";
    }
    outputTarget << "Kernel duration (" << getTimeUnitTag(timeUnit) << "),Global size,Local size";

    auto parameters = results.at(0).getConfiguration().getParameterValues();
    if (parameters.size() > 0)
    {
        outputTarget << ",";
    }

    for (size_t i = 0; i < parameters.size(); i++)
    {
        outputTarget << std::get<0>(parameters.at(i));
        if (i + 1 != parameters.size())
        {
            outputTarget << ",";
        }
    }
    outputTarget << std::endl;

    // Values
    for (const auto& result : results)
    {
        if (!result.isValid())
        {
            continue;
        }

        outputTarget << result.getKernelName() << ",";
        if (results.at(0).getManipulatorDuration() != 0)
        {
            outputTarget << convertTime(result.getTotalDuration(), timeUnit) << ",";
        }
        outputTarget << convertTime(result.getKernelDuration(), timeUnit) << ",";
        printConfigurationCsv(outputTarget, result.getConfiguration());
    }

    if (printInvalidResult)
    {
        outputTarget << std::endl;

        // Header
        outputTarget << "Kernel name,Status,Global size,Local size";

        auto parameters = results.at(0).getConfiguration().getParameterValues();
        if (parameters.size() > 0)
        {
            outputTarget << ",";
        }

        for (size_t i = 0; i < parameters.size(); i++)
        {
            outputTarget << std::get<0>(parameters.at(i));
            if (i + 1 != parameters.size())
            {
                outputTarget << ",";
            }
        }
        outputTarget << std::endl;

        // Values
        for (const auto& result : results)
        {
            if (result.isValid())
            {
                continue;
            }

            outputTarget << result.getKernelName() << ",";
            std::string statusMessage = result.getStatusMessage();
            for (size_t i = 0; i < statusMessage.length(); i++)
            {
                if (statusMessage[i] == '\n' || statusMessage[i] == ',')
                {
                    statusMessage[i] = ' ';
                }
            }

            outputTarget << statusMessage << ",";
            printConfigurationCsv(outputTarget, result.getConfiguration());
        }
    }
}

void ResultPrinter::printConfigurationVerbose(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration) const
{
    DimensionVector convertedGlobalSize = kernelConfiguration.getGlobalSize();
    DimensionVector localSize = kernelConfiguration.getLocalSize();
    if (globalSizeType == GlobalSizeType::Cuda)
    {
        convertedGlobalSize = DimensionVector(std::get<0>(convertedGlobalSize) / std::get<0>(localSize), std::get<1>(convertedGlobalSize)
            / std::get<1>(localSize), std::get<2>(convertedGlobalSize) / std::get<2>(localSize));
    }

    outputTarget << "global size: " << std::get<0>(convertedGlobalSize) << ", " << std::get<1>(convertedGlobalSize) << ", "
        << std::get<2>(convertedGlobalSize) << "; ";
    outputTarget << "local size: " << std::get<0>(localSize) << ", " << std::get<1>(localSize) << ", " << std::get<2>(localSize) << "; ";
    outputTarget << "parameters: ";

    if (kernelConfiguration.getParameterValues().size() == 0)
    {
        outputTarget << "none";
    }
    for (const auto& value : kernelConfiguration.getParameterValues())
    {
        outputTarget << std::get<0>(value) << ": " << std::get<1>(value) << " ";
    }
    outputTarget << std::endl;
}

void ResultPrinter::printConfigurationCsv(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration) const
{
    DimensionVector global = kernelConfiguration.getGlobalSize();
    DimensionVector local = kernelConfiguration.getLocalSize();

    size_t globalSum = std::get<0>(global) * std::get<1>(global) * std::get<2>(global);
    size_t localSum = std::get<0>(local) * std::get<1>(local) * std::get<2>(local);

    if (globalSizeType == GlobalSizeType::Cuda)
    {
        globalSum /= localSum;
    }

    outputTarget << globalSum << ",";
    outputTarget << localSum;

    auto parameterValues = kernelConfiguration.getParameterValues();
    if (parameterValues.size() > 0)
    {
        outputTarget << ",";
    }

    for (size_t i = 0; i < parameterValues.size(); i++)
    {
        outputTarget << std::get<1>(parameterValues.at(i));
        if (i + 1 != parameterValues.size())
        {
            outputTarget << ",";
        }
    }
    outputTarget << std::endl;
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

uint64_t ResultPrinter::convertTime(const uint64_t timeInNanoseconds, const TimeUnit& targetUnit)
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

std::string ResultPrinter::getTimeUnitTag(const TimeUnit& timeUnit)
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
