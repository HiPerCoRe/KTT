#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include <cuda.h>
#include <api/kernel_profiling_data.h>
#include <compute_engine/cuda/cuda_profiling_metric.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUDAProfilingState
{
public:
    explicit CUDAProfilingState(CUcontext context, const CUdevice device, std::vector<std::pair<std::string, CUpti_MetricID>>& metrics) :
        kernelDuration(std::numeric_limits<uint64_t>::max()),
        kernelDurationValid(false),
        remainingKernelRuns(1),
        totalKernelRuns(1),
        eventGroupSets(nullptr),
        currentSetIndex(0)
    {
        std::vector<CUpti_MetricID> metricIds;
        for (const auto& metric : metrics)
        {
            metricIds.push_back(metric.second);
        }

        checkCUDAError(cuptiMetricCreateEventGroupSets(context, sizeof(CUpti_MetricID) * metricIds.size(), metricIds.data(), &eventGroupSets),
            "cuptiMetricCreateEventGroupSets");
        totalKernelRuns += static_cast<uint64_t>(eventGroupSets->numSets);
        remainingKernelRuns += static_cast<uint64_t>(eventGroupSets->numSets);

        for (auto& metric : metrics)
        {
            CUDAProfilingMetric profilingMetric;
            profilingMetric.metricId = metric.second;
            profilingMetric.metricName = metric.first;
            profilingMetric.device = device;
            profilingMetric.eventGroupSets = eventGroupSets;
            checkCUDAError(cuptiMetricGetNumEvents(metric.second, &profilingMetric.eventCount), "cuptiMetricGetNumEvents");
            profilingMetric.eventIds.resize(static_cast<size_t>(profilingMetric.eventCount));
            size_t eventIdsSize = sizeof(CUpti_EventID) * profilingMetric.eventIds.size();
            checkCUDAError(cuptiMetricEnumEvents(metric.second, &eventIdsSize, profilingMetric.eventIds.data()), "cuptiMetricEnumEvents");
            profilingMetric.eventValues.resize(static_cast<size_t>(profilingMetric.eventCount));
            profilingMetric.eventStatuses.resize(static_cast<size_t>(profilingMetric.eventCount), false);
            profilingMetrics.push_back(profilingMetric);
        }
    }

    void updateState()
    {
        if (remainingKernelRuns == 0)
        {
            return;
        }

        ++currentSetIndex;
        for (auto& metric : profilingMetrics)
        {
            metric.currentSetIndex = currentSetIndex;
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

    uint64_t getRemainingKernelRuns() const
    {
        return remainingKernelRuns;
    }

    uint64_t getTotalKernelRuns() const
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

    KernelProfilingData generateProfilingData()
    {
        if (remainingKernelRuns > 0)
        {
            throw std::runtime_error("Internal CUDA CUPTI error : Insufficient number of completed kernel runs for profiling counter collection");
        }

        if (!kernelDurationValid)
        {
            throw std::runtime_error("Internal CUDA CUPTI error : Kernel duration must be known before profiling information can be generated");
        }

        KernelProfilingData result;

        for (auto& metric : profilingMetrics)
        {
            KernelProfilingCounter counter = getCounterFromMetric(metric, kernelDuration);
            result.addCounter(counter);
        }

        return result;
    }

private:
    uint64_t kernelDuration;
    bool kernelDurationValid;
    uint64_t remainingKernelRuns;
    uint64_t totalKernelRuns;
    CUpti_EventGroupSets* eventGroupSets;
    uint32_t currentSetIndex;
    std::vector<CUDAProfilingMetric> profilingMetrics;

    static KernelProfilingCounter getCounterFromMetric(CUDAProfilingMetric& metric, const uint64_t kernelDuration)
    {
        for (const auto eventStatus : metric.eventStatuses)
        {
            if (!eventStatus)
            {
                throw std::runtime_error("Internal CUDA CUPTI error : Failed to collect some metric events for profiling counter calculation");
            }
        }

        CUpti_MetricValue metricValue;
        checkCUDAError(cuptiMetricGetValue(metric.device, metric.metricId, metric.eventCount * sizeof(CUpti_EventID), metric.eventIds.data(),
            metric.eventCount * sizeof(uint64_t), metric.eventValues.data(), kernelDuration, &metricValue), "cuptiMetricGetValue");

        CUpti_MetricValueKind valueKind;
        size_t valueKindSize = sizeof(valueKind);
        checkCUDAError(cuptiMetricGetAttribute(metric.metricId, CUPTI_METRIC_ATTR_VALUE_KIND, &valueKindSize, &valueKind),
            "cuptiMetricGetAttribute");

        ProfilingCounterValue value;
        ProfilingCounterType type;

        switch (valueKind)
        {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
            value.doubleValue = metricValue.metricValueDouble;
            type = ProfilingCounterType::Double;
            break;
        case CUPTI_METRIC_VALUE_KIND_UINT64:
            value.uintValue = metricValue.metricValueUint64;
            type = ProfilingCounterType::UnsignedInt;
            break;
        case CUPTI_METRIC_VALUE_KIND_INT64:
            value.intValue = metricValue.metricValueInt64;
            type = ProfilingCounterType::Int;
            break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT:
            value.percentValue = metricValue.metricValuePercent;
            type = ProfilingCounterType::Percent;
            break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
            value.throughputValue = metricValue.metricValueThroughput;
            type = ProfilingCounterType::Throughput;
            break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
            value.utilizationLevelValue = static_cast<uint32_t>(metricValue.metricValueUtilizationLevel);
            type = ProfilingCounterType::UtilizationLevel;
            break;
        default:
            throw std::runtime_error("Internal CUDA CUPTI error : Unknown metric value type");
        }

        return KernelProfilingCounter(metric.metricName, value, type);
    }
};

} // namespace ktt
