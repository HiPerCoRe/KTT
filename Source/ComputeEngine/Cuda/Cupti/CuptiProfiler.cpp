#if defined(KTT_PROFILING_CUPTI)

#include <cupti_profiler_target.h>

#include <Api/KttException.h>
#include <ComputeEngine/Cuda/Cupti/CuptiProfiler.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CuptiProfiler::CuptiProfiler() :
    m_Counters(GetDefaultCounters())
{
    Logger::LogDebug("Initializing CUPTI profiler");

    CUpti_Profiler_Initialize_Params params =
    {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
        nullptr
    };

    CheckError(cuptiProfilerInitialize(&params), "cuptiProfilerInitialize");
}

CuptiProfiler::~CuptiProfiler()
{
    Logger::LogDebug("Releasing CUPTI profiler");

    CUpti_Profiler_DeInitialize_Params params =
    {
        CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE,
        nullptr
    };

    CheckError(cuptiProfilerDeInitialize(&params), "cuptiProfilerDeInitialize");
}

void CuptiProfiler::SetCounters(const std::vector<std::string>& counters)
{
    if (counters.empty())
    {
        throw KttException("Number of profiling counters must be greater than zero");
    }

    m_Counters = counters;
}

const std::vector<std::string>& CuptiProfiler::GetCounters() const
{
    return m_Counters;
}

const std::vector<std::string>& CuptiProfiler::GetDefaultCounters()
{
    static const std::vector<std::string> result
    {
        "dram__sectors_read.sum", // dram_read_transactions
        "dram__sectors_write.sum", // dram_write_transactions
        "dram__throughput.avg.pct_of_peak_sustained_elapsed", // dram_utilization
        "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed", // shared_utilization
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", // shared_load_transactions
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", // shared_store_transactions
        "lts__t_sectors.avg.pct_of_peak_sustained_elapsed", // l2_utilization
        "sm__warps_active.avg.pct_of_peak_sustained_active", // achieved_occupancy
        "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed", // sm_efficiency
        "smsp__inst_executed_pipe_fp16.sum", // half_precision_fu_utilization
        "smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active", // double_precision_fu_utilization
        "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active", // ldst_fu_utilization
        "smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active", // tex_fu_utilization
        "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active", // special_fu_utilization
        "smsp__inst_executed.sum", // inst_executed
        "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active", // single_precision_fu_utilization
        "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum", // inst_fp_16
        "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum", // inst_fp_32
        "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum", // inst_fp_64
        "smsp__sass_thread_inst_executed_op_integer_pred_on.sum", // inst_integer
        "smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum", // inst_inter_thread_communication
        "smsp__sass_thread_inst_executed_op_misc_pred_on.sum" // inst_misc
    };

    return result;
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
