#pragma once

#if defined(KTT_PROFILING_CUPTI_LEGACY)

#include <ComputeEngine/Cuda/CuptiLegacy/CuptiInstance.h>
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiKtt.h>

namespace ktt
{

class CuptiSubscription
{
public:
    explicit CuptiSubscription(CuptiInstance& instance);
    ~CuptiSubscription();

private:
    CuptiInstance& m_Instance;
    CUpti_SubscriberHandle m_Subscriber;

    static void CUPTIAPI MetricCallback(void* data, CUpti_CallbackDomain domain, CUpti_CallbackId id, const CUpti_CallbackData* info);
    static void BeginCollection(CuptiInstance& instance, const CUpti_CallbackData& info);
    static void EndCollection(CuptiInstance& instance);
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI_LEGACY
