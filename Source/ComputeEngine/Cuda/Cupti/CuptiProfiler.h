#pragma once

#if defined(KTT_PROFILING_CUPTI)

#include <string>
#include <vector>

namespace ktt
{

class CuptiProfiler
{
public:
    CuptiProfiler();
    ~CuptiProfiler();

    void SetCounters(const std::vector<std::string>& counters);
    const std::vector<std::string>& GetCounters() const;

private:
    std::vector<std::string> m_Counters;

    static const std::vector<std::string>& GetDefaultCounters();
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
