#pragma once

#if defined(KTT_POWER_USAGE_NVML)

#include <ComputeEngine/Cuda/CuptiLegacy/CuptiKtt.h>

namespace ktt
{

class NvmlPowerManager;

class NvmlPowerSubscription
{
public:
    explicit NvmlPowerSubscription(NvmlPowerManager& powerManager);
    ~NvmlPowerSubscription();

private:
    NvmlPowerManager& m_PowerManager;
    CUpti_SubscriberHandle m_Subscriber;

    static void CUPTIAPI CollectPowerCallback(void* data, CUpti_CallbackDomain domain, CUpti_CallbackId id,
        const CUpti_CallbackData* info);
};

} // namespace ktt

#endif // KTT_POWER_USAGE_NVML
