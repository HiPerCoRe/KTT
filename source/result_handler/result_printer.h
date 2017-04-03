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
        if (results.find(kernelId) == results.end())
        {
            throw std::runtime_error(std::string("No tuning results found for kernel with id: " + std::to_string(kernelId)));
        }

        auto result = results.find(kernelId)->second;
        switch (printFormat)
        {
        case PrintFormat::CSV:
            printCSV(result, outputTarget);
            break;
        default:
            printVerbose(result, outputTarget);
        }
    }

    void setResult(const size_t kernelId, const std::vector<TuningResult>& result)
    {
        if (results.find(kernelId) != results.end())
        {
            results.erase(kernelId);
        }
        results.insert(std::make_pair(kernelId, result));
    }

private:
    std::map<size_t, std::vector<TuningResult>> results;

    void printVerbose(const std::vector<TuningResult>& result, std::ostream& outputTarget) const
    {
        for (const auto& element : result)
        {
            outputTarget << element;
        }
    }

    void printCSV(const std::vector<TuningResult>& result, std::ostream& outputTarget) const
    {
        // Header
        outputTarget << "Kernel name;Time (us);Threads;Global size;Local size;";
        auto parameters = result.at(0).getConfiguration().getParameterValues();
        for (const auto& parameter : parameters)
        {
            outputTarget << std::get<0>(parameter) << ";";
        }
        outputTarget << std::endl;

        // Values
        for (const auto& element : result)
        {
            outputTarget << element.getKernelName() << ";" << element.getDuration() / 1000 << ";";
            auto configuration = element.getConfiguration();
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
};

} // namespace ktt
