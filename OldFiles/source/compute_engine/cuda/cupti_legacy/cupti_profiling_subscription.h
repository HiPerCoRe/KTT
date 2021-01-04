#pragma once

#include <stdexcept>
#include <vector>
#include <cuda.h>
#include <cupti.h>
#include <compute_engine/cuda/cupti_legacy/cupti_profiling_metric.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUPTIProfilingSubscription
{
public:
    explicit CUPTIProfilingSubscription(std::vector<CUPTIProfilingMetric>& profilingMetrics)
    {
        checkCUPTIError(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &profilingMetrics), "cuptiSubscribe");
        checkCUPTIError(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunch), "cuptiEnableCallback");
        checkCUPTIError(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel), "cuptiEnableCallback");
    }

    ~CUPTIProfilingSubscription()
    {
        checkCUPTIError(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");
    }

private:
    CUpti_SubscriberHandle subscriber;

    static void CUPTIAPI getMetricValueCallback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId id, const CUpti_CallbackData* info)
    {
        if (id != CUPTI_DRIVER_TRACE_CBID_cuLaunch && id != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
        {
            throw std::runtime_error("CUPTI error: Unexpected callback id was passed into metric value collection function");
        }

        auto* metrics = reinterpret_cast<std::vector<CUPTIProfilingMetric>*>(userdata);

        if (metrics->empty())
        {
            return;
        }

        CUPTIProfilingMetric& firstMetric = metrics->at(0);

        if (info->callbackSite == CUPTI_API_ENTER)
        {
            checkCUDAError(cuCtxSynchronize(), "cuCtxSynchronize");
            checkCUPTIError(cuptiSetEventCollectionMode(info->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL), "cuptiSetEventCollectionMode");

            for (uint32_t i = 0; i < firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].numEventGroups; ++i)
            {
                uint32_t profileAll = 1;
                checkCUPTIError(cuptiEventGroupSetAttribute(firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].eventGroups[i],
                    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(profileAll), &profileAll), "cuptiEventGroupSetAttribute");
                checkCUPTIError(cuptiEventGroupEnable(firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].eventGroups[i]),
                    "cuptiEventGroupEnable");
            }
        }
        else if (info->callbackSite == CUPTI_API_EXIT)
        {
            checkCUDAError(cuCtxSynchronize(), "cuCtxSynchronize");

            for (uint32_t i = 0; i < firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].numEventGroups; ++i)
            {
                CUpti_EventGroup group = firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].eventGroups[i];
                CUpti_EventDomainID groupDomain;
                uint32_t eventCount;
                uint32_t instanceCount;
                uint32_t totalInstanceCount;
                size_t groupDomainSize = sizeof(groupDomain);
                size_t eventCountSize = sizeof(eventCount);
                size_t instanceCountSize = sizeof(instanceCount);
                size_t totalInstanceCountSize = sizeof(totalInstanceCount);

                checkCUPTIError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &groupDomainSize, &groupDomain),
                    "cuptiEventGroupGetAttribute");
                checkCUPTIError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &eventCountSize, &eventCount),
                    "cuptiEventGroupGetAttribute");
                checkCUPTIError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &instanceCountSize, &instanceCount),
                    "cuptiEventGroupGetAttribute");
                checkCUPTIError(cuptiDeviceGetEventDomainAttribute(firstMetric.device, groupDomain, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                    &totalInstanceCountSize, &totalInstanceCount), "cuptiDeviceGetEventDomainAttribute");

                std::vector<CUpti_EventID> eventIds(eventCount);
                size_t eventIdsSize = eventCount * sizeof(CUpti_EventID);
                checkCUPTIError(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &eventIdsSize, eventIds.data()),
                    "cuptiEventGroupGetAttribute");

                std::vector<uint64_t> values(instanceCount);
                size_t valuesSize = instanceCount * sizeof(uint64_t);

                for (uint32_t j = 0; j < eventCount; ++j)
                {
                    checkCUPTIError(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE, eventIds[j], &valuesSize, values.data()),
                        "cuptiEventGroupReadEvent");

                    uint64_t sum = 0;
                    for (uint32_t k = 0; k < instanceCount; ++k)
                    {
                        sum += values[k];
                    }

                    const uint64_t normalized = (sum * totalInstanceCount) / instanceCount;
                    for (auto& metric : *metrics)
                    {
                        for (size_t k = 0; k < metric.eventIds.size(); ++k)
                        {
                            if (metric.eventIds[k] == eventIds[j] && !metric.eventStatuses[k])
                            {
                                metric.eventValues[k] = normalized;
                                metric.eventStatuses[k] = true;
                            }
                        }
                    }
                }
            }

            for (uint32_t i = 0; i < firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].numEventGroups; ++i)
            {
                checkCUPTIError(cuptiEventGroupDisable(firstMetric.eventGroupSets->sets[firstMetric.currentSetIndex].eventGroups[i]),
                    "cuptiEventGroupDisable");
            }
        }
    }
};

} // namespace ktt
