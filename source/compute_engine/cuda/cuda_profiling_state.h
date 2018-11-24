#pragma once

#include <limits>
#include <string>
#include <vector>
#include <cuda.h>
#include <compute_engine/cuda/cuda_profiling_metric.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUDAProfilingState
{
public:
    explicit CUDAProfilingState(CUcontext context, const CUdevice device, std::vector<CUpti_MetricID>& metricIds) :
        kernelDuration(std::numeric_limits<uint64_t>::max()),
        kernelDurationValid(false),
        eventGroups(nullptr),
        remainingKernelRuns(0),
        totalKernelRuns(0)
    {
        checkCUDAError(cuptiMetricCreateEventGroupSets(context, metricIds.size(), metricIds.data(), &eventGroups),
            "cuptiMetricCreateEventGroupSets");
        totalKernelRuns = eventGroups->numSets + 1;
        remainingKernelRuns = eventGroups->numSets + 1;

        for (const auto metricId : metricIds)
        {
            CUDAProfilingMetric metric;
            metric.device = device;
            metric.currentSet = &eventGroups->sets[0];
            checkCUDAError(cuptiMetricGetNumEvents(metricId, &metric.eventCount), "cuptiMetricGetNumEvents");
            metric.eventIds.resize(static_cast<size_t>(metric.eventCount));
            metric.eventValues.resize(static_cast<size_t>(metric.eventCount));
            profilingMetrics.push_back(metric);
        }
    }

    void updateState()
    {
        for (auto& metric : profilingMetrics)
        {
            metric.currentSet = &eventGroups->sets[totalKernelRuns - remainingKernelRuns - 1];
        }
        --remainingKernelRuns;
    }

    void updateState(const uint64_t kernelDuration)
    {
        this->kernelDuration = kernelDuration;
        kernelDurationValid = true;
        --remainingKernelRuns;
    }

    uint64_t getKernelDuration() const
    {
        return kernelDuration;
    }

    bool hasValidKernelDuration() const
    {
        return kernelDurationValid;
    }

    CUpti_EventGroupSets* getEventGroups() const
    {
        return eventGroups;
    }

    size_t getRemainingKernelRuns() const
    {
        return remainingKernelRuns;
    }

    size_t getTotalKernelRuns() const
    {
        return totalKernelRuns;
    }

    std::vector<CUDAProfilingMetric>* getProfilingMetrics()
    {
        return &profilingMetrics;
    }

    const std::vector<CUDAProfilingMetric>& getProfilingMetrics() const
    {
        return profilingMetrics;
    }

private:
    uint64_t kernelDuration;
    bool kernelDurationValid;
    CUpti_EventGroupSets* eventGroups;
    size_t remainingKernelRuns;
    size_t totalKernelRuns;
    std::vector<CUDAProfilingMetric> profilingMetrics;
};

} // namespace ktt
