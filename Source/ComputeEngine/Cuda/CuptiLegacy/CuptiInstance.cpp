#if defined(KTT_PROFILING_CUPTI_LEGACY)

#include <algorithm>
#include <limits>
#include <string>

#include <Api/Output/KernelProfilingData.h>
#include <Api/KttException.h>
#include <ComputeEngine/Cuda/CuptiLegacy/CuptiInstance.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CuptiInstance::CuptiInstance(const CudaContext& context) :
    m_Context(context),
    m_KernelDuration(InvalidDuration),
    m_CurrentPassIndex(0),
    m_EventSets(nullptr)
{
    Logger::LogDebug("Initializing CUPTI instance");

    if (m_EnabledMetrics.empty())
    {
        SetEnabledMetrics(GetDefaultMetrics());
    }

    std::vector<CUpti_MetricID> metricIds;

    for (const auto& metric : m_EnabledMetrics)
    {
        CUpti_MetricID id;
        const auto result = cuptiMetricGetIdFromName(context.GetDevice(), metric.c_str(), &id);

        if (result == CUPTI_ERROR_INVALID_METRIC_NAME)
        {
            Logger::LogWarning("Metric with name " + metric + " is not supported on the current device");
            continue;
        }

        CheckError(result, "cuptiMetricGetIdFromName");
        metricIds.push_back(id);
        m_Metrics.push_back(std::make_unique<CuptiMetric>(id, metric));
    }

    if (metricIds.empty())
    {
        throw KttException("No valid metrics provided for the current device");
    }

    CheckError(cuptiMetricCreateEventGroupSets(context.GetContext(), sizeof(CUpti_MetricID) * metricIds.size(), metricIds.data(),
        &m_EventSets), "cuptiMetricCreateEventGroupSets");
    m_TotalPassCount = static_cast<uint64_t>(m_EventSets->numSets);
}

CuptiInstance::~CuptiInstance()
{
    Logger::LogDebug("Releasing CUPTI instance");
    CheckError(cuptiEventGroupSetsDestroy(m_EventSets), "cuptiEventGroupSetsDestroy");
}

void CuptiInstance::UpdatePassIndex()
{
    ++m_CurrentPassIndex;
}

void CuptiInstance::SetKernelDuration(const Nanoseconds duration)
{
    KttAssert(duration != InvalidDuration, "Kernel duration must be valid");
    m_KernelDuration = duration;
}

const CudaContext& CuptiInstance::GetContext() const
{
    return m_Context;
}

Nanoseconds CuptiInstance::GetKernelDuration() const
{
    return m_KernelDuration;
}

bool CuptiInstance::HasValidKernelDuration() const
{
    return m_KernelDuration != InvalidDuration;
}

uint64_t CuptiInstance::GetRemainingPassCount() const
{
    if (m_CurrentPassIndex > m_TotalPassCount)
    {
        return 0;
    }

    return m_TotalPassCount - m_CurrentPassIndex;
}

uint64_t CuptiInstance::GetTotalPassCount() const
{
    return m_TotalPassCount;
}

uint32_t CuptiInstance::GetCurrentIndex() const
{
    return static_cast<uint32_t>(m_CurrentPassIndex);
}

CUpti_EventGroupSets& CuptiInstance::GetEventSets()
{
    return *m_EventSets;
}

std::vector<std::unique_ptr<CuptiMetric>>& CuptiInstance::GetMetrics()
{
    return m_Metrics;
}

std::unique_ptr<KernelProfilingData> CuptiInstance::GenerateProfilingData()
{
    KttAssert(HasValidKernelDuration(), "Kernel duration must be known before profiling information can be generated");
    const uint64_t remainingCount = GetRemainingPassCount();

    if (remainingCount > 0)
    {
        return std::make_unique<KernelProfilingData>(remainingCount);
    }

    Logger::LogDebug("Generating profiling data for CUPTI instance");
    std::vector<KernelProfilingCounter> counters;

    for (auto& metric : m_Metrics)
    {
        KernelProfilingCounter counter = metric->GenerateCounter(m_Context, m_KernelDuration);
        counters.push_back(counter);
    }

    return std::make_unique<KernelProfilingData>(counters);
}

void CuptiInstance::SetEnabledMetrics(const std::vector<std::string>& metrics)
{
    if (metrics.empty())
    {
        throw KttException("Number of profiling metrics must be greater than zero");
    }

    m_EnabledMetrics = metrics;
}

const std::vector<std::string>& CuptiInstance::GetDefaultMetrics()
{
    static const std::vector<std::string> result
    {
        "dram_utilization",
        "dram_read_transactions",
        "dram_write_transactions",
        "l2_utilization",
        "l2_read_transactions",
        "l2_write_transactions",
        "tex_utilization",
        "tex_cache_transactions",
        "local_memory_overhead",
        "shared_utilization",
        "shared_efficiency",         /* shared_utilization equivalent for cc 3.x */
        "shared_load_transactions",
        "shared_store_transactions",
        "local_load_transactions",
        "local_store_transactions",
        "achieved_occupancy",
        "sm_efficiency",
        "inst_fp_16",
        "inst_fp_32",
        "inst_fp_64",
        "inst_integer",
        "inst_control",
        "inst_compute_ld_st",
        "inst_misc",
        "inst_bit_convert",
        "inst_executed",
        "half_precision_fu_utilization",
        "single_precision_fu_utilization",
        "flop_sp_efficiency",       /* single_precision_fu_utilization equivalent for cc 3.x */
        "double_precision_fu_utilization",
        "flop_dp_efficiency",       /* double_precision_fu_utilization equivalent for cc 3.x */
        "special_fu_utilization",
        "cf_fu_utilization",
        "ldst_fu_utilization",
        "tex_fu_utilization",
        "issue_slot_utilization",
        "warp_execution_efficiency",
        "warp_nonpred_execution_efficiency"
    };

    return result;
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI_LEGACY
