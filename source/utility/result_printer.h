#pragma once

#include <map>
#include <string>
#include <vector>

#include "dto/tuning_result.h"
#include "enum/print_format.h"
#include "enum/time_unit.h"

namespace ktt
{

class ResultPrinter
{
public:
    ResultPrinter();

    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void setResult(const size_t kernelId, const std::vector<TuningResult>& result, const std::vector<TuningResult>& invalidResult);
    void setTimeUnit(const TimeUnit& timeUnit);
    void setInvalidResultPrinting(const bool flag);

private:
    std::map<size_t, std::vector<TuningResult>> resultMap;
    std::map<size_t, std::vector<TuningResult>> invalidResultMap;
    TimeUnit timeUnit;
    bool printInvalidResult;

    void printVerbose(const std::vector<TuningResult>& results, const std::vector<TuningResult>& invalidResults, std::ostream& outputTarget) const;
    void printCsv(const std::vector<TuningResult>& results, const std::vector<TuningResult>& invalidResults, std::ostream& outputTarget) const;
    TuningResult getBestResult(const std::vector<TuningResult>& results) const;
    uint64_t convertTime(const uint64_t timeInNanoseconds, const TimeUnit& targetUnit) const;
    std::string getTimeUnitTag(const TimeUnit& timeUnit) const;
};

} // namespace ktt
