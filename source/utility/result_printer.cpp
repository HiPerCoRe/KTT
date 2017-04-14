#include "result_printer.h"

namespace ktt
{

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

void ResultPrinter::printVerbose(const std::vector<TuningResult>& results, std::ostream& outputTarget) const
{
    for (const auto& result : results)
    {
        outputTarget << result << std::endl;
    }

    auto bestResult = getBestResult(results);
    outputTarget << "Best result: " << std::endl;
    outputTarget << bestResult;
}

void ResultPrinter::printCSV(const std::vector<TuningResult>& results, std::ostream& outputTarget) const
{
    // Header
    outputTarget << "Kernel name;Total duration (us);Kernel duration (us);Global size;Local size;Threads;";
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

        outputTarget << result.getKernelName() << ";" << result.getTotalDuration() / 1000 << ";" << result.getKernelDuration() / 1000 << ";";
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

} // namespace ktt
