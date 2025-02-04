#if defined(KTT_POWER_USAGE_NVML)

#include <ComputeEngine/Cuda/Nvml/NvmlPowerManager.h>
#include <ComputeEngine/Cuda/Nvml/NvmlPowerSubscription.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

NvmlPowerSubscription::NvmlPowerSubscription(NvmlPowerManager& powerManager) :
    m_PowerManager(powerManager)
{
    Logger::LogDebug("Activating power data collection subscription");
    CheckError(cuptiSubscribe(&m_Subscriber, (CUpti_CallbackFunc)NvmlPowerSubscription::CollectPowerCallback, &powerManager),
        "cuptiSubscribe");
    CheckError(cuptiEnableCallback(1, m_Subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel),
        "cuptiEnableCallback");
}

NvmlPowerSubscription::~NvmlPowerSubscription()
{
    Logger::LogDebug("Deactivating power data collection subscription");
    CheckError(cuptiUnsubscribe(m_Subscriber), "cuptiUnsubscribe");
}

void NvmlPowerSubscription::CollectPowerCallback(void* data, [[maybe_unused]]  CUpti_CallbackDomain domain,
    [[maybe_unused]] CUpti_CallbackId id, const CUpti_CallbackData* info)
{
    KttAssert(id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, "Unexpected callback id was passed into power info collection function");
    auto& powerManager = *reinterpret_cast<NvmlPowerManager*>(data);

    if (info->callbackSite == CUPTI_API_ENTER)
    {
        powerManager.StartCollection();
    }
    else if (info->callbackSite == CUPTI_API_EXIT)
    {
        powerManager.EndCollection();
    }
}

} // namespace ktt

#endif // KTT_POWER_USAGE_NVML
