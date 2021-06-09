#pragma once

#if defined(KTT_PROFILING_CUPTI)

namespace ktt
{

class CuptiProfiler
{
public:
    CuptiProfiler();
    ~CuptiProfiler();
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
