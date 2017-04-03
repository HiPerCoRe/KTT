#pragma once

#include <map>
#include <vector>

#include "../enum/print_format.h"
#include "../dto/tuning_result.h"

namespace ktt
{

class ResultPrinter
{
public:
    ResultPrinter() = default;

    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const
    {
        if (resultMap.find(kernelId) == resultMap.end())
        {
            throw std::runtime_error(std::string("No tuning results found for kernel with id: " + std::to_string(kernelId)));
        }

        auto results = resultMap.find(kernelId)->second;
        if (results.size() == 0)
        {
            throw std::runtime_error(std::string("No tuning results found for kernel with id: " + std::to_string(kernelId)));
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

    void setResult(const size_t kernelId, const std::vector<TuningResult>& result)
    {
        if (resultMap.find(kernelId) != resultMap.end())
        {
            resultMap.erase(kernelId);
        }
        resultMap.insert(std::make_pair(kernelId, result));
    }

private:
    std::map<size_t, std::vector<TuningResult>> resultMap;

    void printVerbose(const std::vector<TuningResult>& results, std::ostream& outputTarget) const
    {
        for (const auto& result : results)
        {
            outputTarget << result << std::endl;
        }

        auto bestResult = getBestResult(results);
        outputTarget << "Best result: " << std::endl;
        outputTarget << bestResult;
    }

    void printCSV(const std::vector<TuningResult>& results, std::ostream& outputTarget) const
    {
        // Header
        outputTarget << "Kernel name;Time (us);Threads;Global size;Local size;";
        auto parameters = results.at(0).getConfiguration().getParameterValues();
        for (const auto& parameter : parameters)
        {
            outputTarget << std::get<0>(parameter) << ";";
        }
        outputTarget << std::endl;

        // Values
        for (const auto& result : results)
        {
            outputTarget << result.getKernelName() << ";" << result.getDuration() / 1000 << ";";
            auto configuration = result.getConfiguration();
            auto global = configuration.getGlobalSize();
            outputTarget << std::get<0>(global) * std::get<1>(global) * std::get<2>(global) << ";";
            outputTarget << std::get<0>(global) << " " << std::get<1>(global) << " " << std::get<2>(global) << ";";
            auto local = configuration.getLocalSize();
            outputTarget << std::get<0>(local) << " " << std::get<1>(local) << " " << std::get<2>(local) << ";";

            auto parameterValues = configuration.getParameterValues();
            for (const auto& value : parameterValues)
            {
                outputTarget << std::get<1>(value) << ";";
            }
            outputTarget << std::endl;
        }
    }

    TuningResult getBestResult(const std::vector<TuningResult>& results) const
    {
        TuningResult bestResult = results.at(0);

        for (const auto& result : results)
        {
            if (result.getDuration() < bestResult.getDuration())
            {
                bestResult = result;
            }
        }

        return bestResult;
    }
};

} // namespace ktt
