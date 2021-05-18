#ifdef KTT_PROFILING_CUPTI

#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

#include <Api/Output/KernelProfilingData.h>
#include <Api/KttException.h>
#include <ComputeEngine/Cuda/Cupti/CuptiMetricInterface.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

CuptiMetricInterface::CuptiMetricInterface(const DeviceIndex index) :
    m_DeviceName(GetDeviceName(index)),
    m_Context(nullptr),
    m_MaxProfiledRanges(2),
    m_MaxRangeNameLength(64)
{
    Logger::LogDebug("Initializing CUPTI metric interface");

    NVPW_InitializeHost_Params hostParams =
    {
        NVPW_InitializeHost_Params_STRUCT_SIZE,
        nullptr
    };

    CheckError(NVPW_InitializeHost(&hostParams), "NVPW_InitializeHost");

    NVPW_CUDA_MetricsContext_Create_Params params =
    {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
        nullptr,
        m_DeviceName.data(),
        nullptr
    };

    CheckError(NVPW_CUDA_MetricsContext_Create(&params), "NVPW_CUDA_MetricsContext_Create");
    m_Context = params.pMetricsContext;
    SetMetrics(GetDefaultMetrics());
}

CuptiMetricInterface::~CuptiMetricInterface()
{
    Logger::LogDebug("Releasing CUPTI metric interface");

    NVPW_MetricsContext_Destroy_Params params =
    {
        NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
        nullptr,
        m_Context
    };

    CheckError(NVPW_MetricsContext_Destroy(&params), "NVPW_MetricsContext_Destroy");
}

void CuptiMetricInterface::SetMetrics(const std::vector<std::string>& metrics)
{
    if (metrics.empty())
    {
        throw KttException("Number of profiling metrics must be greater than zero");
    }

    std::vector<std::string> filteredMetrics;
    const auto supportedMetrics = GetSupportedMetrics(true);

    for (const auto& metric : metrics)
    {
        if (!ContainsKey(supportedMetrics, metric))
        {
            Logger::LogWarning("Metric with name " + metric + " is not supported on the current device");
            continue;
        }

        filteredMetrics.push_back(metric);
    }

    m_Metrics = filteredMetrics;
}

CuptiMetricConfiguration CuptiMetricInterface::CreateMetricConfiguration() const
{
    CuptiMetricConfiguration result(m_MaxProfiledRanges);
    result.m_MetricNames = m_Metrics;
    result.m_ConfigImage = GetConfigImage(m_Metrics);
    std::vector<uint8_t> prefix = GetCounterDataImagePrefix(m_Metrics);
    CreateCounterDataImage(prefix, result.m_CounterDataImage, result.m_ScratchBuffer);
    return result;
}

std::unique_ptr<KernelProfilingData> CuptiMetricInterface::GenerateProfilingData(const CuptiMetricConfiguration& configuration) const
{
    if (!configuration.m_DataCollected)
    {
        return std::make_unique<KernelProfilingData>(1);
    }

    const auto& counterDataImage = configuration.m_CounterDataImage;

    NVPW_CounterData_GetNumRanges_Params params =
    {
        NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE,
        nullptr,
        counterDataImage.data(),
        0
    };

    CheckError(NVPW_CounterData_GetNumRanges(&params), "NVPW_CounterData_GetNumRanges");

    const auto& metricNames = configuration.m_MetricNames;
    std::vector<std::string> parsedNames(metricNames.size());
    std::vector<const char*> metricNamePtrs;
    bool isolated = true;
    bool keepInstances = true;

    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
    {
        [[maybe_unused]] const bool success = ParseMetricNameString(metricNames[metricIndex], parsedNames[metricIndex], isolated,
            keepInstances);
        KttAssert(success, "Unable to parse metric name " + metricNames[metricIndex]);
        metricNamePtrs.push_back(parsedNames[metricIndex].c_str());
    }

    std::vector<KernelProfilingCounter> counters;

    for (size_t rangeIndex = 0; rangeIndex < params.numRanges; ++rangeIndex)
    {
        NVPW_MetricsContext_SetCounterData_Params dataParams =
        {
            NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE,
            nullptr,
            m_Context,
            counterDataImage.data(),
            rangeIndex,
            isolated
        };

        CheckError(NVPW_MetricsContext_SetCounterData(&dataParams), "NVPW_MetricsContext_SetCounterData");
        std::vector<double> gpuValues(metricNames.size());

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalParams =
        {
            NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE,
            nullptr,
            m_Context,
            metricNamePtrs.size(),
            metricNamePtrs.data(),
            gpuValues.data()
        };

        CheckError(NVPW_MetricsContext_EvaluateToGpuValues(&evalParams), "NVPW_MetricsContext_EvaluateToGpuValues");

        if (rangeIndex > 0)
        {
            // Only values from the first range are currently utilized for counters
            continue;
        }

        for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
        {
            counters.emplace_back(metricNames[metricIndex], ProfilingCounterType::Double, gpuValues[metricIndex]);
        }
    }

    return std::make_unique<KernelProfilingData>(counters);
}

void CuptiMetricInterface::ListSupportedChips()
{
    NVPW_GetSupportedChipNames_Params params =
    {
        NVPW_GetSupportedChipNames_Params_STRUCT_SIZE,
        nullptr,
        nullptr,
        0
    };

    CheckError(NVPW_GetSupportedChipNames(&params), "NVPW_GetSupportedChipNames");
    std::string chips;

    for (size_t i = 0; i < params.numChipNames; ++i)
    {
        chips += params.ppChipNames[i];

        if (i + 1 != params.numChipNames)
        {
            chips += ", ";
        }
    }

    Logger::LogInfo("Number of supported chips for CUPTI profiling: " + std::to_string(params.numChipNames));
    Logger::LogInfo("List of supported chips: " + chips);
}

std::set<std::string> CuptiMetricInterface::GetSupportedMetrics(const bool listSubMetrics) const
{
    NVPW_MetricsContext_GetMetricNames_Begin_Params params =
    {
        NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE,
        nullptr,
        m_Context,
        0,
        nullptr,
        !listSubMetrics,
        !listSubMetrics,
        !listSubMetrics,
        !listSubMetrics,
    };

    CheckError(NVPW_MetricsContext_GetMetricNames_Begin(&params), "NVPW_MetricsContext_GetMetricNames_Begin");
    std::set<std::string> result;

    for (size_t i = 0; i < params.numMetrics; ++i)
    {
        result.insert(params.ppMetricNames[i]);
    }

    NVPW_MetricsContext_GetMetricNames_End_Params endParams =
    {
        NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE,
        nullptr,
        m_Context
    };

    CheckError(NVPW_MetricsContext_GetMetricNames_End(&endParams), "NVPW_MetricsContext_GetMetricNames_End");
    return result;
}

std::vector<uint8_t> CuptiMetricInterface::GetConfigImage(const std::vector<std::string>& metrics) const
{
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string> temp;
    GetRawMetricRequests(metrics, rawMetricRequests, temp);

    NVPA_RawMetricsConfigOptions configOptions =
    {
        NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE,
        nullptr,
        NVPA_ACTIVITY_KIND_PROFILER,
        m_DeviceName.c_str()
    };

    NVPA_RawMetricsConfig* rawMetricsConfig;
    CheckError(NVPA_RawMetricsConfig_Create(&configOptions, &rawMetricsConfig), "NVPA_RawMetricsConfig_Create");

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginParams =
    {
        NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig
    };

    CheckError(NVPW_RawMetricsConfig_BeginPassGroup(&beginParams), "NVPW_RawMetricsConfig_BeginPassGroup");

    NVPW_RawMetricsConfig_AddMetrics_Params addParams =
    {
        NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig,
        rawMetricRequests.data(),
        rawMetricRequests.size()
    };

    CheckError(NVPW_RawMetricsConfig_AddMetrics(&addParams), "NVPW_RawMetricsConfig_AddMetrics");

    NVPW_RawMetricsConfig_EndPassGroup_Params endParams =
    {
        NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig
    };

    CheckError(NVPW_RawMetricsConfig_EndPassGroup(&endParams), "NVPW_RawMetricsConfig_EndPassGroup");

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateParams =
    {
        NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig
    };

    CheckError(NVPW_RawMetricsConfig_GenerateConfigImage(&generateParams), "NVPW_RawMetricsConfig_GenerateConfigImage");

    NVPW_RawMetricsConfig_GetConfigImage_Params getParams =
    {
        NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig,
        0,
        nullptr,
        0
    };

    CheckError(NVPW_RawMetricsConfig_GetConfigImage(&getParams), "NVPW_RawMetricsConfig_GetConfigImage");

    std::vector<uint8_t> result(getParams.bytesCopied);
    getParams.bytesAllocated = result.size();
    getParams.pBuffer = result.data();

    CheckError(NVPW_RawMetricsConfig_GetConfigImage(&getParams), "NVPW_RawMetricsConfig_GetConfigImage");

    NVPW_RawMetricsConfig_Destroy_Params destroyParams =
    {
        NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig
    };

    CheckError(NVPW_RawMetricsConfig_Destroy(&destroyParams), "NVPW_RawMetricsConfig_Destroy");
    return result;
}

std::vector<uint8_t> CuptiMetricInterface::GetCounterDataImagePrefix(const std::vector<std::string>& metrics) const
{
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string> temp;
    GetRawMetricRequests(metrics, rawMetricRequests, temp);

    NVPW_CounterDataBuilder_Create_Params createParams =
    {
        NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE,
        nullptr,
        nullptr,
        m_DeviceName.c_str()
    };

    CheckError(NVPW_CounterDataBuilder_Create(&createParams), "NVPW_CounterDataBuilder_Create");

    NVPW_CounterDataBuilder_AddMetrics_Params addParams =
    {
        NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE,
        nullptr,
        createParams.pCounterDataBuilder,
        rawMetricRequests.data(),
        rawMetricRequests.size()
    };

    CheckError(NVPW_CounterDataBuilder_AddMetrics(&addParams), "NVPW_CounterDataBuilder_AddMetrics");

    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getParams =
    {
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
        nullptr,
        createParams.pCounterDataBuilder,
        0,
        nullptr,
        0
    };

    CheckError(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getParams), "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    std::vector<uint8_t> result(getParams.bytesCopied);
    getParams.bytesAllocated = result.size();
    getParams.pBuffer = result.data();

    CheckError(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getParams), "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    NVPW_CounterDataBuilder_Destroy_Params destroyParams =
    {
        NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE,
        nullptr,
        createParams.pCounterDataBuilder
    };

    CheckError(NVPW_CounterDataBuilder_Destroy(&destroyParams), "NVPW_CounterDataBuilder_Destroy");
    return result;
}

void CuptiMetricInterface::CreateCounterDataImage(const std::vector<uint8_t>& counterDataImagePrefix,
    std::vector<uint8_t>& counterDataImage, std::vector<uint8_t>& counterDataScratchBuffer) const
{
    const CUpti_Profiler_CounterDataImageOptions counterDataImageOptions =
    {
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        nullptr,
        counterDataImagePrefix.data(),
        counterDataImagePrefix.size(),
        m_MaxProfiledRanges,
        m_MaxProfiledRanges,
        m_MaxRangeNameLength
    };

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams =
    {
        CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
        nullptr,
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        &counterDataImageOptions,
        0
    };

    CheckError(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams), "cuptiProfilerCounterDataImageCalculateSize");
    counterDataImage.resize(calculateSizeParams.counterDataImageSize);

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams =
    {
        CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
        nullptr,
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        &counterDataImageOptions,
        counterDataImage.size(),
        counterDataImage.data()
    };

    CheckError(cuptiProfilerCounterDataImageInitialize(&initializeParams), "cuptiProfilerCounterDataImageInitialize");

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchSizeParams =
    {
        CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        nullptr,
        counterDataImage.size(),
        counterDataImage.data(),
        0
    };

    CheckError(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchSizeParams),
        "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    counterDataScratchBuffer.resize(scratchSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initializeScratchParams =
    {
        CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
        nullptr,
        counterDataImage.size(),
        counterDataImage.data(),
        counterDataScratchBuffer.size(),
        counterDataScratchBuffer.data()
    };

    CheckError(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initializeScratchParams),
        "cuptiProfilerCounterDataImageInitializeScratchBuffer");
}

void CuptiMetricInterface::GetRawMetricRequests(const std::vector<std::string>& metrics,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests, std::vector<std::string>& temp) const
{
    bool isolated = true;
    bool keepInstances = true;

    for (const auto& metricName : metrics)
    {
        std::string parsedName;
        [[maybe_unused]] const bool success = ParseMetricNameString(metricName, parsedName, isolated, keepInstances);
        KttAssert(success, "Unable to parse metric name " + metricName);

        keepInstances = true; // Bug in collection with collection of metrics without instances, keep it to true

        NVPW_MetricsContext_GetMetricProperties_Begin_Params params =
        {
            NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
            nullptr,
            m_Context,
            parsedName.c_str(),
            nullptr,
            nullptr,
            nullptr,
            0.0,
            0.0
        };

        CheckError(NVPW_MetricsContext_GetMetricProperties_Begin(&params), "NVPW_MetricsContext_GetMetricProperties_Begin");

        for (const char** metricDependencies = params.ppRawMetricDependencies; *metricDependencies != nullptr; ++metricDependencies)
        {
            temp.push_back(*metricDependencies);
        }

        NVPW_MetricsContext_GetMetricProperties_End_Params endParams =
        {
            NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
            nullptr,
            m_Context
        };

        CheckError(NVPW_MetricsContext_GetMetricProperties_End(&endParams), "NVPW_MetricsContext_GetMetricProperties_End");
    }

    for (const auto& rawMetricName : temp)
    {
        NVPA_RawMetricRequest request =
        {
            NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE,
            nullptr,
            rawMetricName.c_str(),
            isolated,
            keepInstances
        };

        rawMetricRequests.push_back(request);
    }
}

const std::vector<std::string>& CuptiMetricInterface::GetDefaultMetrics()
{
    static const std::vector<std::string> result
    {
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__sectors_read.sum",
        "dram__sectors_write.sum",
        "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "smsp__sass_thread_inst_executed_op_fp16_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fp64_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_control_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_memory_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_misc_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum",
        "smsp__inst_executed.sum",
        "smsp__inst_executed_pipe_fp16.sum",
        "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
        "smsp__thread_inst_executed_per_inst_executed.ratio",
        "smsp__thread_inst_executed_per_inst_executed.pct",
    };

    return result;
}

std::string CuptiMetricInterface::GetDeviceName(const DeviceIndex index)
{
    CUpti_Device_GetChipName_Params params =
    {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE,
        nullptr,
        static_cast<size_t>(index),
        nullptr
    };

    CheckError(cuptiDeviceGetChipName(&params), "cuptiDeviceGetChipName");
    return params.pChipName;
}

bool CuptiMetricInterface::ParseMetricNameString(const std::string& metric, std::string& outputName, bool& isolated,
    bool& keepInstances)
{
    if (metric.empty())
    {
        outputName = "";
        return false;
    }

    outputName = metric;
    keepInstances = false;
    isolated = true;

    // boost program_options sometimes inserts a \n between the metric name and a '&' at the end
    size_t pos = outputName.find('\n');

    if (pos != std::string::npos)
    {
        outputName.erase(pos, 1);
    }

    // trim whitespace
    while (outputName.back() == ' ')
    {
        outputName.pop_back();

        if (outputName.empty())
        {
            return false;
        }
    }

    if (outputName.back() == '+')
    {
        keepInstances = true;
        outputName.pop_back();

        if (outputName.empty())
        {
            return false;
        }
    }

    if (outputName.back() == '$')
    {
        outputName.pop_back();

        if (outputName.empty())
        {
            return false;
        }
    }
    else if (outputName.back() == '&')
    {
        isolated = false;
        outputName.pop_back();

        if (outputName.empty())
        {
            return false;
        }
    }

    return true;
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
