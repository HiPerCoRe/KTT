#pragma once

#include <map>
#include <string>
#include <vector>

#include "dto/tuning_result.h"
#include "enum/global_size_type.h"
#include "enum/print_format.h"
#include "enum/time_unit.h"

namespace ktt
{

class ResultPrinter
{
public:
    ResultPrinter();

    void printResult(const size_t kernelId, std::ostream& outputTarget, const PrintFormat& printFormat) const;
    void setResult(const size_t kernelId, const std::vector<TuningResult>& results);
    void setTimeUnit(const TimeUnit& timeUnit);
    void setInvalidResultPrinting(const bool flag);
    std::vector<ParameterValue> getBestConfiguration(const size_t kernelId) const;

private:
    std::map<size_t, std::vector<TuningResult>> resultMap;
    TimeUnit timeUnit;
    bool printInvalidResult;

    void printVerbose(const std::vector<TuningResult>& results, std::ostream& outputTarget) const;
    void printCsv(const std::vector<TuningResult>& results, std::ostream& outputTarget) const;
    void printConfigurationVerbose(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration) const;
    void printConfigurationCsv(std::ostream& outputTarget, const KernelConfiguration& kernelConfiguration) const;
    TuningResult getBestResult(const std::vector<TuningResult>& results) const;
    static uint64_t convertTime(const uint64_t timeInNanoseconds, const TimeUnit& targetUnit);
    static std::string getTimeUnitTag(const TimeUnit& timeUnit);
};

} // namespace ktt
