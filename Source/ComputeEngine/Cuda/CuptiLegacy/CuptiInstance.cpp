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
#if 0
        /* minimalistic counters configurations needed for profile-based searcher */
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
#else
        /*  exhaustive profile counters list */
        "achieved_occupancy",
        "atomic_transactions",
        "atomic_transactions_per_request",
        "branch_efficiency",
        "cf_executed",
        "cf_fu_utilization",
        "cf_issued",
        "double_precision_fu_utilization",
        "flop_dp_efficiency",       /* double_precision_fu_utilization equivalent for cc 3.x */
        "dram_read_bytes",
        "dram_read_throughput",
        "dram_read_transactions",
        "dram_utilization",
        "dram_write_bytes",
        "dram_write_throughput",
        "dram_write_transactions",
        "eligible_warps_per_cycle",
        "flop_count_sp",
        "flop_count_sp_add",
        "flop_count_sp_fma",
        "flop_count_sp_mul",
        "flop_count_sp_special",
        "flop_sp_efficiency",
        "gld_efficiency",
        "gld_requested_throughput",
        "gld_throughput",
        "gld_transactions",
        "gld_transactions_per_request",
        "global_atomic_requests",
        "global_hit_rate",
        "global_load_requests",
        "global_reduction_requests",
        "global_store_requests",
        "gst_efficiency",
        "gst_requested_throughput",
        "gst_throughput",
        "gst_transactions",
        "gst_transactions_per_request",
        "inst_bit_convert",
        "inst_compute_ld_st",
        "inst_control",
        "inst_executed",
        "inst_executed_global_atomics",
        "inst_executed_global_loads",
        "inst_executed_global_reductions",
        "inst_executed_global_stores",
        "inst_executed_local_loads",
        "inst_executed_local_stores",
        "inst_executed_shared_atomics",
        "inst_executed_shared_loads",
        "inst_executed_shared_stores",
        "inst_executed_surface_atomics",
        "inst_executed_surface_loads",
        "inst_executed_surface_reductions",
        "inst_executed_surface_stores",
        "inst_executed_tex_ops",
        "inst_fp_32",
        "inst_integer",
        "inst_inter_thread_communication",
        "inst_issued",
        "inst_misc",
        "inst_per_warp",
        "inst_replay_overhead",
        "ipc",
        "issue_slot_utilization",
        "issue_slots",
        "issued_ipc",
        "l2_atomic_throughput",
        "l2_atomic_transactions",
        "l2_global_atomic_store_bytes",
        "l2_global_load_bytes",
        "l2_global_reduction_bytes",
        "l2_local_global_store_bytes",
        "l2_local_load_bytes",
        "l2_read_throughput",
        "l2_read_transactions",
        "l2_surface_atomic_store_bytes",
        "l2_surface_load_bytes",
        "l2_surface_reduction_bytes",
        "l2_surface_store_bytes",
        "l2_tex_hit_rate",
        "l2_tex_read_hit_rate",
        "l2_tex_read_throughput",
        "l2_tex_read_transactions",
        "l2_tex_write_hit_rate",
        "l2_tex_write_throughput",
        "l2_tex_write_transactions",
        "l2_utilization",
        "l2_write_throughput",
        "l2_write_transactions",
        "ldst_executed",
        "ldst_fu_utilization",
        "ldst_issued",
        "local_hit_rate",
        "local_load_requests",
        "local_load_throughput",
        "local_load_transactions",
        "local_load_transactions_per_request",
        "local_memory_overhead",
        "local_store_requests",
        "local_store_throughput",
        "local_store_transactions",
        "local_store_transactions_per_request",
        "pcie_total_data_received",
        "pcie_total_data_transmitted",
        "shared_efficiency",
        "shared_load_throughput",
        "shared_load_transactions",
        "shared_load_transactions_per_request",
        "shared_store_throughput",
        "shared_store_transactions",
        "shared_store_transactions_per_request",
        "shared_utilization",
        "shared_efficiency",         /* shared_utilization equivalent for cc 3.x */
        "single_precision_fu_utilization",
        "flop_sp_efficiency",       /* single_precision_fu_utilization equivalent for cc 3.x */
        "sm_efficiency",
        "special_fu_utilization",
        "stall_constant_memory_dependency",
        "stall_exec_dependency",
        "stall_inst_fetch",
        "stall_memory_dependency",
        "stall_memory_throttle",
        "stall_not_selected",
        "stall_other",
        "stall_pipe_busy",
        "stall_sync",
        "stall_texture",
        "surface_atomic_requests",
        "surface_load_requests",
        "surface_reduction_requests",
        "surface_store_requests",
        "tex_cache_hit_rate",
        "tex_cache_throughput",
        "tex_cache_transactions",
        "tex_fu_utilization",
        "tex_utilization",
        "texture_load_requests",
        "warp_execution_efficiency",
        "warp_nonpred_execution_efficiency"
#endif
    };

    return result;
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI_LEGACY
