#pragma once

#ifdef KTT_PROFILING_CUPTI

#include <map>
#include <string>

#include <Api/Output/KernelProfilingCounter.h>

namespace ktt
{

class CuptiMetric
{
public:
    CuptiMetric(const std::string& name);

    void SetRangeValue(const std::string& range, const double value);
    KernelProfilingCounter GenerateCounter() const;
    void PrintData() const;

private:
    std::string m_Name;
    std::map<std::string, double> m_RangeToValue;
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
