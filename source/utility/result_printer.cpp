#include <algorithm>
#include "result_printer.h"

namespace ktt
{

ResultPrinter::ResultPrinter() :
    timeUnit(TimeUnit::Microseconds),
    printInvalidResult(false)
{}

void ResultPrinter::printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat& format) const
{
    if (kernelResults.find(id) == kernelResults.end())
    {
        throw std::runtime_error(std::string("No tuning results found for kernel with id: ") + std::to_string(id));
    }

    std::vector<TuningResult> results = kernelResults.find(id)->second;

    switch (format)
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

void ResultPrinter::setResult(const KernelId id, const std::vector<TuningResult>& results)
{
    if (kernelResults.find(id) != kernelResults.end())
    {
        kernelResults.erase(id);
    }

    kernelResults.insert(std::make_pair(id, results));
}

void ResultPrinter::setTimeUnit(const TimeUnit& unit)
{
    this->timeUnit = unit;
}

void ResultPrinter::setInvalidResultPrinting(const TunerFlag flag)
{
    printInvalidResult = flag;
}

std::vector<ParameterPair> ResultPrinter::getBestConfiguration(const KernelId id) const
{
    if (kernelResults.find(id) == kernelResults.end())
    {
        throw std::runtime_error(std::string("No tuning results found for kernel with id: ") + std::to_string(id));
    }

    std::vector<TuningResult> results = kernelResults.find(id)->second;
    TuningResult bestResult = getBestResult(results);
    return bestResult.getConfiguration().getParameterPairs();
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

    TuningResult bestResult = getBestResult(results);
    if (bestResult.isValid())
    {
        outputTarget << "Best result for kernel <" << bestResult.getKernelName() << ">: " << std::endl;
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
    outputTarget << "Kernel duration (" << getTimeUnitTag(timeUnit) << ")";

    size_t kernelCount = results.at(0).getConfiguration().getGlobalSizes().size();
    if (kernelCount == 1)
    {
        outputTarget << ",Global size,Local size";
    }
    else
    {
        for (size_t i = 0; i < kernelCount; i++)
        {
            outputTarget << ",Global size " << i << ",Local size " << i;
        }
    }

    std::vector<ParameterPair> parameterPairs = results.at(0).getConfiguration().getParameterPairs();
    if (parameterPairs.size() > 0)
    {
        outputTarget << ",";
    }

    for (size_t i = 0; i < parameterPairs.size(); i++)
    {
        outputTarget << std::get<0>(parameterPairs.at(i));
        if (i + 1 != parameterPairs.size())
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
        outputTarget << "Kernel name,Status";

        if (kernelCount == 1)
        {
            outputTarget << ",Global size,Local size";
        }
        else
        {
            for (size_t i = 0; i < kernelCount; i++)
            {
                outputTarget << ",Global size " << i << ",Local size " << i;
            }
        }

        std::vector<ParameterPair> parameterPairs = results.at(0).getConfiguration().getParameterPairs();
        if (parameterPairs.size() > 0)
        {
            outputTarget << ",";
        }

        for (size_t i = 0; i < parameterPairs.size(); i++)
        {
            outputTarget << std::get<0>(parameterPairs.at(i));
            if (i + 1 != parameterPairs.size())
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

void ResultPrinter::printConfigurationVerbose(std::ostream& outputTarget, const KernelConfiguration& configuration) const
{
    std::vector<DimensionVector> globalSizes = configuration.getGlobalSizes();
    std::vector<DimensionVector> localSizes = configuration.getLocalSizes();

    for (size_t i = 0; i < globalSizes.size(); i++)
    {
        DimensionVector convertedGlobalSize = globalSizes.at(i);
        DimensionVector localSize = localSizes.at(i);

        if (configuration.getGlobalSizeType() == GlobalSizeType::Cuda)
        {
            convertedGlobalSize.divide(localSize);
        }

        if (globalSizes.size() > 1)
        {
            outputTarget << "global size " << i << ": " << convertedGlobalSize << "; ";
            outputTarget << "local size " << i << ": " << localSize << "; ";
        }
        else
        {
            outputTarget << "global size: " << convertedGlobalSize << "; ";
            outputTarget << "local size: " << localSize << "; ";
        }
    }

    outputTarget << "parameters: ";
    if (configuration.getParameterPairs().size() == 0)
    {
        outputTarget << "none";
    }
    for (const auto& parameterPair : configuration.getParameterPairs())
    {
        outputTarget << std::get<0>(parameterPair) << ": " << std::get<1>(parameterPair) << " ";
    }
    outputTarget << std::endl;
}

void ResultPrinter::printConfigurationCsv(std::ostream& outputTarget, const KernelConfiguration& configuration) const
{
    std::vector<DimensionVector> globalSizes = configuration.getGlobalSizes();
    std::vector<DimensionVector> localSizes = configuration.getLocalSizes();

    for (size_t i = 0; i < globalSizes.size(); i++)
    {
        DimensionVector globalSize = globalSizes.at(i);
        DimensionVector localSize = localSizes.at(i);

        size_t totalGlobalSize = globalSize.getTotalSize();
        size_t totalLocalSize = localSize.getTotalSize();

        if (configuration.getGlobalSizeType() == GlobalSizeType::Cuda)
        {
            totalGlobalSize /= totalLocalSize;
        }

        outputTarget << totalGlobalSize << ",";
        outputTarget << totalLocalSize;

        if (i + 1 != globalSizes.size())
        {
            outputTarget << ",";
        }
    }

    std::vector<ParameterPair> parameterPairs = configuration.getParameterPairs();
    if (parameterPairs.size() > 0)
    {
        outputTarget << ",";
    }

    for (size_t i = 0; i < parameterPairs.size(); i++)
    {
        outputTarget << std::get<1>(parameterPairs.at(i));
        if (i + 1 != parameterPairs.size())
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

std::string ResultPrinter::getTimeUnitTag(const TimeUnit& unit)
{
    switch (unit)
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
