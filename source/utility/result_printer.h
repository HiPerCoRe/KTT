#pragma once

#include <map>
#include <string>
#include <vector>
#include "dto/kernel_result.h"
#include "enum/print_format.h"
#include "enum/time_unit.h"

namespace ktt
{

class ResultPrinter
{
public:
    ResultPrinter();

    void printResult(const KernelId id, std::ostream& outputTarget, const PrintFormat format) const;
    void addResult(const KernelId id, const KernelResult& result);
    void setResult(const KernelId id, const std::vector<KernelResult>& results);
    void setTimeUnit(const TimeUnit unit);
    void setInvalidResultPrinting(const bool flag);
    void clearResults(const KernelId id);

private:
    std::map<KernelId, std::vector<KernelResult>> kernelResults;
    TimeUnit timeUnit;
    bool printInvalidResult;

    void printVerbose(const std::vector<KernelResult>& results, std::ostream& outputTarget) const;
    void printCsv(const std::vector<KernelResult>& results, std::ostream& outputTarget) const;
    void printConfigurationVerbose(std::ostream& outputTarget, const KernelConfiguration& configuration) const;
    void printConfigurationCsv(std::ostream& outputTarget, const KernelConfiguration& configuration) const;
    KernelResult getBestResult(const std::vector<KernelResult>& results) const;
    static uint64_t convertTime(const uint64_t timeInNanoseconds, const TimeUnit targetUnit);
    static std::string getTimeUnitTag(const TimeUnit unit);
};

} // namespace ktt
