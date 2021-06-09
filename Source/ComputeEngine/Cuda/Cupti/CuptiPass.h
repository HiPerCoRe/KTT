#pragma once

#ifdef KTT_PROFILING_CUPTI

#include <ComputeEngine/Cuda/Cupti/CuptiInstance.h>

namespace ktt
{

class CuptiPass
{
public:
    explicit CuptiPass(CuptiInstance& instance);
    ~CuptiPass();

private:
    CuptiInstance& m_Instance;
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
