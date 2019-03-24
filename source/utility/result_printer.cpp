#include <algorithm>
#include <utility/ktt_utility.h>
#include <utility/result_printer.h>

namespace ktt
{

ResultPrinter::ResultPrinter() :
    timeUnit(TimeUnit::Milliseconds),
    printInvalidResult(false)
{}

void ResultPrinter::printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const
{
    if (kernelResults.find(id) == kernelResults.end())
    {
        throw std::runtime_error(std::string("No tuning results found for kernel with id: ") + std::to_string(id));
    }

    std::vector<KernelResult> results = kernelResults.find(id)->second;

    switch (format)
    {
    case PrintFormat::CSV:
        printCSV(results, outputTarget);
        break;
    case PrintFormat::Verbose:
        printVerbose(results, outputTarget);
        break;
    default:
        throw std::runtime_error("Unknown print format");
    }
}

void ResultPrinter::addResult(const KernelId id, const KernelResult& result)
{
    if (kernelResults.find(id) == kernelResults.end())
    {
        kernelResults.insert(std::make_pair(id, std::vector<KernelResult>{}));
    }

    kernelResults.find(id)->second.push_back(result);
}

void ResultPrinter::setResult(const KernelId id, const std::vector<KernelResult>& results)
{
    if (kernelResults.find(id) != kernelResults.end())
    {
        kernelResults.erase(id);
    }

    kernelResults.insert(std::make_pair(id, results));
}

void ResultPrinter::setTimeUnit(const TimeUnit unit)
{
    this->timeUnit = unit;
}

void ResultPrinter::setInvalidResultPrinting(const bool flag)
{
    printInvalidResult = flag;
}

void ResultPrinter::clearResults(const KernelId id)
{
    if (kernelResults.find(id) != kernelResults.end())
    {
        kernelResults.erase(id);
    }
}

void ResultPrinter::printVerbose(const std::vector<KernelResult>& results, std::ostream& outputTarget) const
{
    for (const auto& result : results)
    {
        if (!result.isValid())
        {
            continue;
        }

        outputTarget << "Result for kernel " << result.getKernelName() << ", configuration: " << std::endl;
        printConfigurationVerbose(outputTarget, result.getConfiguration());
        outputTarget << "Computation duration: " << convertTime(result.getComputationDuration(), timeUnit) << getTimeUnitTag(timeUnit);
        outputTarget << std::endl;
    }

    KernelResult bestResult = getBestResult(results);
    if (bestResult.isValid())
    {
        outputTarget << "Best result for kernel " << bestResult.getKernelName() << ": " << std::endl;
        outputTarget << "Configuration: ";
        printConfigurationVerbose(outputTarget, bestResult.getConfiguration());
        outputTarget << "Computation duration: " << convertTime(bestResult.getComputationDuration(), timeUnit) << getTimeUnitTag(timeUnit);
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

            outputTarget << "Invalid result for kernel " << result.getKernelName() << ", configuration: " << std::endl;
            printConfigurationVerbose(outputTarget, result.getConfiguration());
            outputTarget << "Error message: " << result.getErrorMessage();
            outputTarget << std::endl;
        }
    }
}

void ResultPrinter::printCSV(const std::vector<KernelResult>& results, std::ostream& outputTarget) const
{
    // Header
    outputTarget << "Kernel name," << "Computation duration (" << getTimeUnitTag(timeUnit) << ")";

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
        outputTarget << parameterPairs.at(i).getName();
        if (i + 1 != parameterPairs.size())
        {
            outputTarget << ",";
        }
    }

    if (results.at(0).getProfilingData().isValid())
    {
        const std::vector<KernelProfilingCounter>& counters = results.at(0).getProfilingData().getAllCounters();
        if (counters.size() > 0)
        {
            outputTarget << ",";
        }

        for (size_t i = 0; i < counters.size(); ++i)
        {
            outputTarget << counters.at(i).getName();
            if (i + 1 != counters.size())
            {
                outputTarget << ",";
            }
        }
    }
    else if (!results.at(0).getCompositionProfilingData().empty())
    {
        for (const auto& pair : results.at(0).getCompositionProfilingData())
        {
            if (!pair.second.isValid())
            {
                continue;
            }

            const std::vector<KernelProfilingCounter>& counters = pair.second.getAllCounters();
            if (counters.size() > 0)
            {
                outputTarget << ",";
            }

            for (size_t i = 0; i < counters.size(); ++i)
            {
                outputTarget << counters.at(i).getName() << " " << pair.first;
                if (i + 1 != counters.size())
                {
                    outputTarget << ",";
                }
            }
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
        outputTarget << convertTime(result.getComputationDuration(), timeUnit) << ",";
        printConfigurationCSV(outputTarget, result.getConfiguration(), parameterPairs);
        if (result.getProfilingData().isValid())
        {
            printProfilingCountersCSV(outputTarget, result.getProfilingData().getAllCounters());
        }
        else if (!result.getCompositionProfilingData().empty())
        {
            for (const auto& pair : result.getCompositionProfilingData())
            {
                if (!pair.second.isValid())
                {
                    continue;
                }

                printProfilingCountersCSV(outputTarget, pair.second.getAllCounters());
            }
        }
        outputTarget << std::endl;
    }

    if (printInvalidResult)
    {
        outputTarget << std::endl;

        // Header
        outputTarget << "Kernel name,Error message";

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
            outputTarget << parameterPairs.at(i).getName();
            if (i + 1 != parameterPairs.size())
            {
                outputTarget << ",";
            }
        }

        if (results.at(0).getProfilingData().isValid())
        {
            const std::vector<KernelProfilingCounter>& counters = results.at(0).getProfilingData().getAllCounters();
            if (counters.size() > 0)
            {
                outputTarget << ",";
            }

            for (size_t i = 0; i < counters.size(); ++i)
            {
                outputTarget << counters.at(i).getName();
                if (i + 1 != counters.size())
                {
                    outputTarget << ",";
                }
            }
        }
        else if (!results.at(0).getCompositionProfilingData().empty())
        {
            for (const auto& pair : results.at(0).getCompositionProfilingData())
            {
                if (!pair.second.isValid())
                {
                    continue;
                }

                const std::vector<KernelProfilingCounter>& counters = pair.second.getAllCounters();
                if (counters.size() > 0)
                {
                    outputTarget << ",";
                }

                for (size_t i = 0; i < counters.size(); ++i)
                {
                    outputTarget << counters.at(i).getName() << " " << pair.first;
                    if (i + 1 != counters.size())
                    {
                        outputTarget << ",";
                    }
                }
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
            std::string statusMessage = result.getErrorMessage();
            for (size_t i = 0; i < statusMessage.length(); i++)
            {
                if (statusMessage[i] == '\n' || statusMessage[i] == ',')
                {
                    statusMessage[i] = ' ';
                }
            }

            outputTarget << statusMessage << ",";
            printConfigurationCSV(outputTarget, result.getConfiguration(), parameterPairs);
            if (result.getProfilingData().isValid())
            {
                printProfilingCountersCSV(outputTarget, result.getProfilingData().getAllCounters());
            }
            else if (!result.getCompositionProfilingData().empty())
            {
                for (const auto& pair : result.getCompositionProfilingData())
                {
                    if (!pair.second.isValid())
                    {
                        continue;
                    }

                    printProfilingCountersCSV(outputTarget, pair.second.getAllCounters());
                }
            }
            outputTarget << std::endl;
        }
    }
}

void ResultPrinter::printConfigurationVerbose(std::ostream& outputTarget, const KernelConfiguration& configuration) const
{
    std::vector<DimensionVector> globalSizes = configuration.getGlobalSizes();
    std::vector<DimensionVector> localSizes = configuration.getLocalSizes();

    for (size_t i = 0; i < globalSizes.size(); i++)
    {
        DimensionVector globalSize = globalSizes.at(i);
        DimensionVector localSize = localSizes.at(i);

        if (globalSizes.size() > 1)
        {
            outputTarget << "global size " << i << ": " << globalSize << "; ";
            outputTarget << "local size " << i << ": " << localSize << "; ";
        }
        else
        {
            outputTarget << "global size: " << globalSize << "; ";
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
        outputTarget << parameterPair << " ";
    }
    outputTarget << std::endl;
}

void ResultPrinter::printConfigurationCSV(std::ostream& outputTarget, const KernelConfiguration& configuration,
    const std::vector<ParameterPair>& orderedPairs) const
{
    std::vector<DimensionVector> globalSizes = configuration.getGlobalSizes();
    std::vector<DimensionVector> localSizes = configuration.getLocalSizes();

    for (size_t i = 0; i < globalSizes.size(); i++)
    {
        DimensionVector globalSize = globalSizes.at(i);
        DimensionVector localSize = localSizes.at(i);

        size_t totalGlobalSize = globalSize.getTotalSize();
        size_t totalLocalSize = localSize.getTotalSize();

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
        size_t pairIndex = 0;
        for (size_t j = 0; j < parameterPairs.size(); j++)
        {
            if (parameterPairs.at(j).getName() == orderedPairs.at(i).getName())
            {
                pairIndex = j;
                break;
            }
        }

        if (!parameterPairs.at(pairIndex).hasValueDouble())
        {
            outputTarget << parameterPairs.at(pairIndex).getValue();
        }
        else
        {
            outputTarget << parameterPairs.at(pairIndex).getValueDouble();
        }

        if (i + 1 != parameterPairs.size())
        {
            outputTarget << ",";
        }
    }
}

void ResultPrinter::printProfilingCountersCSV(std::ostream& outputTarget, const std::vector<KernelProfilingCounter>& counters) const
{
    if (counters.size() > 0)
    {
        outputTarget << ",";
    }

    for (size_t i = 0; i < counters.size(); ++i)
    {
        const KernelProfilingCounter& counter = counters.at(i);
        switch (counter.getType())
        {
        case ProfilingCounterType::Double:
            outputTarget << counter.getValue().doubleValue;
            break;
        case ProfilingCounterType::Int:
            outputTarget << counter.getValue().intValue;
            break;
        case ProfilingCounterType::UnsignedInt:
            outputTarget << counter.getValue().uintValue;
            break;
        case ProfilingCounterType::Percent:
            outputTarget << counter.getValue().percentValue;
            break;
        case ProfilingCounterType::Throughput:
            outputTarget << counter.getValue().throughputValue;
            break;
        case ProfilingCounterType::UtilizationLevel:
            outputTarget << counter.getValue().utilizationLevelValue;
            break;
        default:
            throw std::runtime_error("Unknown profiling counter type");
        }

        if (i + 1 != counters.size())
        {
            outputTarget << ",";
        }
    }
}

KernelResult ResultPrinter::getBestResult(const std::vector<KernelResult>& results) const
{
    KernelResult bestResult = KernelResult();
    bestResult.setComputationDuration(UINT64_MAX);

    for (const auto& result : results)
    {
        if (result.isValid() && result.getComputationDuration() < bestResult.getComputationDuration())
        {
            bestResult = result;
        }
    }

    return bestResult;
}

} // namespace ktt
