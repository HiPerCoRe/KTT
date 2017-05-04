#pragma once

#include <map>
#include <vector>

#include "../dto/tuning_result.h"
#include "../enum/print_format.h"

namespace ktt
{

class ResultPrinter
{
public:
    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void setResult(const size_t kernelId, const std::vector<TuningResult>& result);

private:
    std::map<size_t, std::vector<TuningResult>> resultMap;

    void printVerbose(const std::vector<TuningResult>& results, std::ostream& outputTarget) const;
    void printCSV(const std::vector<TuningResult>& results, std::ostream& outputTarget) const;
    TuningResult getBestResult(const std::vector<TuningResult>& results) const;
};

} // namespace ktt
