#if defined(KTT_PROFILING_CUPTI_LEGACY)

#include <cuda.h>

#include <ComputeEngine/Cuda/CuptiLegacy/CuptiSubscription.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CuptiSubscription::CuptiSubscription(CuptiInstance& instance) :
    m_Instance(instance)
{
    Logger::LogDebug("Activating CUPTI subscription");
    CheckError(cuptiSubscribe(&m_Subscriber, (CUpti_CallbackFunc)CuptiSubscription::MetricCallback, &instance),
        "cuptiSubscribe");
    CheckError(cuptiEnableCallback(1, m_Subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunch),
        "cuptiEnableCallback");
    CheckError(cuptiEnableCallback(1, m_Subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel),
        "cuptiEnableCallback");
}

CuptiSubscription::~CuptiSubscription()
{
    Logger::LogDebug("Deactivating CUPTI subscription");
    CheckError(cuptiUnsubscribe(m_Subscriber), "cuptiUnsubscribe");
    m_Instance.UpdatePassIndex();
}

void CuptiSubscription::MetricCallback(void* data, [[maybe_unused]]  CUpti_CallbackDomain domain,
    [[maybe_unused]] CUpti_CallbackId id, const CUpti_CallbackData* info)
{
    KttAssert(id == CUPTI_DRIVER_TRACE_CBID_cuLaunch || id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
        "Unexpected callback id was passed into metric value collection function");

    auto& instance = *reinterpret_cast<CuptiInstance*>(data);
    KttAssert(instance.GetRemainingPassCount() > 0, "Running metric value collection for instance which has already all data collected");

    if (info->callbackSite == CUPTI_API_ENTER)
    {
        BeginCollection(instance, *info);
    }
    else if (info->callbackSite == CUPTI_API_EXIT)
    {
        EndCollection(instance);
    }
}

void CuptiSubscription::BeginCollection(CuptiInstance& instance, const CUpti_CallbackData& info)
{
    CheckError(cuCtxSynchronize(), "cuCtxSynchronize");
    CheckError(cuptiSetEventCollectionMode(info.context, CUPTI_EVENT_COLLECTION_MODE_KERNEL), "cuptiSetEventCollectionMode");

    auto& sets = instance.GetEventSets();
    const uint32_t index = instance.GetCurrentIndex();

    for (uint32_t i = 0; i < sets.sets[index].numEventGroups; ++i)
    {
        uint32_t profileAll = 1;
        CheckError(cuptiEventGroupSetAttribute(sets.sets[index].eventGroups[i], CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
            sizeof(profileAll), &profileAll), "cuptiEventGroupSetAttribute");
        CheckError(cuptiEventGroupEnable(sets.sets[index].eventGroups[i]), "cuptiEventGroupEnable");
    }
}

void CuptiSubscription::EndCollection(CuptiInstance& instance)
{
    CheckError(cuCtxSynchronize(), "cuCtxSynchronize");

    auto& sets = instance.GetEventSets();
    const uint32_t index = instance.GetCurrentIndex();

    for (uint32_t i = 0; i < sets.sets[index].numEventGroups; ++i)
    {
        CUpti_EventGroup group = sets.sets[index].eventGroups[i];
        CUpti_EventDomainID groupDomain;
        uint32_t eventCount;
        uint32_t instanceCount;
        uint32_t totalInstanceCount;
        size_t groupDomainSize = sizeof(groupDomain);
        size_t eventCountSize = sizeof(eventCount);
        size_t instanceCountSize = sizeof(instanceCount);
        size_t totalInstanceCountSize = sizeof(totalInstanceCount);

        CheckError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize, &groupDomain),
            "cuptiEventGroupGetAttribute");
        CheckError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &eventCountSize, &eventCount),
            "cuptiEventGroupGetAttribute");
        CheckError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &instanceCountSize, &instanceCount),
            "cuptiEventGroupGetAttribute");
        CheckError(cuptiDeviceGetEventDomainAttribute(instance.GetContext().GetDevice(), groupDomain,
            CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,  &totalInstanceCountSize, &totalInstanceCount),
            "cuptiDeviceGetEventDomainAttribute");

        std::vector<CUpti_EventID> eventIds(eventCount);
        size_t eventIdsSize = eventCount * sizeof(CUpti_EventID);
        CheckError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds.data()),
            "cuptiEventGroupGetAttribute");

        std::vector<uint64_t> values(instanceCount);
        size_t valuesSize = instanceCount * sizeof(uint64_t);

        for (uint32_t j = 0; j < eventCount; ++j)
        {
            CheckError(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE, eventIds[j], &valuesSize, values.data()),
                "cuptiEventGroupReadEvent");

            uint64_t sum = 0;

            for (uint32_t k = 0; k < instanceCount; ++k)
            {
                sum += values[k];
            }

            const uint64_t normalized = (sum * totalInstanceCount) / instanceCount;

            for (auto& metric : instance.GetMetrics())
            {
                metric->SetEventValue(eventIds[j], normalized);
            }
        }
    }

    for (uint32_t i = 0; i < sets.sets[index].numEventGroups; ++i)
    {
        CheckError(cuptiEventGroupDisable(sets.sets[index].eventGroups[i]), "cuptiEventGroupDisable");
    }
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI_LEGACY
