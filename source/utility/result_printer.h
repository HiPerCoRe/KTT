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

    void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat& format) const;
    void setResult(const KernelId id, const std::vector<TuningResult>& results);
    void setTimeUnit(const TimeUnit& unit);
    void setInvalidResultPrinting(const TunerFlag flag);
    std::vector<ParameterPair> getBestConfiguration(const KernelId id) const;

private:
    std::map<KernelId, std::vector<TuningResult>> kernelResults;
    TimeUnit timeUnit;
    TunerFlag printInvalidResult;

    void printVerbose(const std::vector<TuningResult>& results, std::ostream& outputTarget) const;
    void printCsv(const std::vector<TuningResult>& results, std::ostream& outputTarget) const;
    void printConfigurationVerbose(std::ostream& outputTarget, const KernelConfiguration& configuration) const;
    void printConfigurationCsv(std::ostream& outputTarget, const KernelConfiguration& configuration) const;
    TuningResult getBestResult(const std::vector<TuningResult>& results) const;
    static uint64_t convertTime(const uint64_t timeInNanoseconds, const TimeUnit& targetUnit);
    static std::string getTimeUnitTag(const TimeUnit& unit);
};

} // namespace ktt
