#ifdef KTT_PROFILING_CUPTI

#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

#include <Api/Output/KernelProfilingData.h>
#include <Api/KttException.h>
#include <ComputeEngine/Cuda/Cupti/CuptiMetricInterface.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

CuptiMetricInterface::CuptiMetricInterface(const DeviceIndex index, const CudaContext& context) :
    m_DeviceName(GetDeviceName(index)),
    m_Evaluator(nullptr),
    m_MaxProfiledRanges(2),
    m_MaxRangeNameLength(64)
{
    Logger::LogDebug("Initializing CUPTI metric interface");
    InitializeCounterAvailabilityImage(context);

    NVPW_InitializeHost_Params hostParams =
    {
        NVPW_InitializeHost_Params_STRUCT_SIZE,
        nullptr
    };

    CheckError(NVPW_InitializeHost(&hostParams), "NVPW_InitializeHost");

    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params scratchBufferSizeParams =
    {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        nullptr,
        m_DeviceName.c_str(),
        m_CounterAvailabilityImage.data(),
        0
    };

    CheckError(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&scratchBufferSizeParams), "NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize");
    m_ScratchBuffer.resize(scratchBufferSizeParams.scratchBufferSize);

    NVPW_CUDA_MetricsEvaluator_Initialize_Params evaluatorInitializeParams =
    {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE,
        nullptr,
        m_ScratchBuffer.data(),
        m_ScratchBuffer.size(),
        m_DeviceName.c_str(),
        m_CounterAvailabilityImage.data(),
        nullptr,
        0,
        nullptr
    };

    CheckError(NVPW_CUDA_MetricsEvaluator_Initialize(&evaluatorInitializeParams), "NVPW_CUDA_MetricsEvaluator_Initialize");
    m_Evaluator = evaluatorInitializeParams.pMetricsEvaluator;
    SetMetrics(GetDefaultMetrics());
}

CuptiMetricInterface::~CuptiMetricInterface()
{
    Logger::LogDebug("Releasing CUPTI metric interface");

    NVPW_MetricsEvaluator_Destroy_Params params =
    {
        NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE,
        nullptr,
        m_Evaluator
    };

    CheckError(NVPW_MetricsEvaluator_Destroy(&params), "NVPW_MetricsEvaluator_Destroy");
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

    if (filteredMetrics.empty())
    {
        throw KttException("No valid metrics provided for the current device");
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
    bool isolated = true;
    bool keepInstances = true;
    std::vector<KernelProfilingCounter> counters;

    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
    {
        std::string parsedName;
        [[maybe_unused]] const bool success = ParseMetricNameString(metricNames[metricIndex], parsedName, isolated, keepInstances);
        KttAssert(success, "Unable to parse metric name " + metricNames[metricIndex]);
        NVPW_MetricEvalRequest evalRequest;

        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params evalRequestParams =
        {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE,
            nullptr,
            m_Evaluator,
            parsedName.c_str(),
            &evalRequest,
            NVPW_MetricEvalRequest_STRUCT_SIZE
        };

        CheckError(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&evalRequestParams),
            "NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest");
        double counterValue = 0.0;

        for (size_t rangeIndex = 0; rangeIndex < params.numRanges; ++rangeIndex)
        {
            NVPW_MetricsEvaluator_SetDeviceAttributes_Params attributesParams =
            {
                NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE,
                nullptr,
                m_Evaluator,
                counterDataImage.data(),
                counterDataImage.size()
            };

            CheckError(NVPW_MetricsEvaluator_SetDeviceAttributes(&attributesParams), "NVPW_MetricsEvaluator_SetDeviceAttributes");
            double metricValue = 0.0;

            NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateParams =
            {
                NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE,
                nullptr,
                m_Evaluator,
                &evalRequest,
                1,
                NVPW_MetricEvalRequest_STRUCT_SIZE,
                sizeof(NVPW_MetricEvalRequest),
                counterDataImage.data(),
                counterDataImage.size(),
                rangeIndex,
                isolated,
                &metricValue
            };

            CheckError(NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateParams), "NVPW_MetricsEvaluator_EvaluateToGpuValues");

            if (rangeIndex == 0)
            {
                // Only values from the first range are currently utilized for counters
                counterValue = metricValue;
            }
        }

        counters.emplace_back(metricNames[metricIndex], ProfilingCounterType::Double, counterValue);
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
    std::set<std::string> result;

    for (int i = 0; i < static_cast<int>(NVPW_MetricType::NVPW_METRIC_TYPE__COUNT); ++i)
    {
        const auto metricType = static_cast<NVPW_MetricType>(i);

        NVPW_MetricsEvaluator_GetMetricNames_Params params =
        {
            NVPW_MetricsEvaluator_GetMetricNames_Params_STRUCT_SIZE,
            nullptr,
            m_Evaluator,
            static_cast<uint8_t>(metricType),
            nullptr,
            nullptr,
            0
        };

        CheckError(NVPW_MetricsEvaluator_GetMetricNames(&params), "NVPW_MetricsEvaluator_GetMetricNames");

        for (size_t metricIndex = 0; metricIndex < params.numMetrics; ++metricIndex)
        {
            size_t metricNameIndex = params.pMetricNameBeginIndices[metricIndex];

            for (int rollupOp = 0; rollupOp < static_cast<int>(NVPW_RollupOp::NVPW_ROLLUP_OP__COUNT); ++rollupOp)
            {
                std::string metricName = &params.pMetricNames[metricNameIndex];

                if (metricType != NVPW_MetricType::NVPW_METRIC_TYPE_RATIO)
                {
                    metricName += GetMetricRollupOpString(static_cast<NVPW_RollupOp>(rollupOp));
                }

                if (!listSubMetrics)
                {
                    result.insert(metricName);
                    continue;
                }

                NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params submetricsParmas =
                {
                    NVPW_MetricsEvaluator_GetSupportedSubmetrics_Params_STRUCT_SIZE,
                    nullptr,
                    m_Evaluator,
                    static_cast<uint8_t>(metricType),
                    nullptr,
                    0
                };

                CheckError(NVPW_MetricsEvaluator_GetSupportedSubmetrics(&submetricsParmas),
                    "NVPW_MetricsEvaluator_GetSupportedSubmetrics");

                for (size_t submetricIndex = 0; submetricIndex < submetricsParmas.numSupportedSubmetrics; ++submetricIndex)
                {
                    const auto submetric = static_cast<NVPW_Submetric>(submetricsParmas.pSupportedSubmetrics[submetricIndex]);
                    std::string submetricName = metricName + GetSubmetricString(submetric);
                    result.insert(submetricName);
                }
            }
        }
    }

    return result;
}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif // __GNUC__

std::vector<uint8_t> CuptiMetricInterface::GetConfigImage(const std::vector<std::string>& metrics) const
{
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    GetRawMetricRequests(metrics, rawMetricRequests);

    NVPW_CUDA_RawMetricsConfig_Create_V2_Params configCreateParams =
    {
        NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE,
        nullptr,
        NVPA_ACTIVITY_KIND_PROFILER,
        m_DeviceName.c_str(),
        m_CounterAvailabilityImage.data(),
        nullptr
    };

    CheckError(NVPW_CUDA_RawMetricsConfig_Create_V2(&configCreateParams), "NVPW_CUDA_RawMetricsConfig_Create_V2");
    NVPA_RawMetricsConfig* rawMetricsConfig = configCreateParams.pRawMetricsConfig;

    if (!m_CounterAvailabilityImage.empty())
    {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params counterAvailabilityParams =
        {
            NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE,
            nullptr,
            rawMetricsConfig,
            m_CounterAvailabilityImage.data()
        };

        CheckError(NVPW_RawMetricsConfig_SetCounterAvailability(&counterAvailabilityParams),
            "NVPW_RawMetricsConfig_SetCounterAvailability");
    }

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

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif // __GNUC__

std::vector<uint8_t> CuptiMetricInterface::GetCounterDataImagePrefix(const std::vector<std::string>& metrics) const
{
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    GetRawMetricRequests(metrics, rawMetricRequests);

    NVPW_CUDA_CounterDataBuilder_Create_Params createParams =
    {
        NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE,
        nullptr,
        m_DeviceName.c_str(),
        m_CounterAvailabilityImage.data(),
        nullptr
    };

    CheckError(NVPW_CUDA_CounterDataBuilder_Create(&createParams), "NVPW_CUDA_CounterDataBuilder_Create");

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
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests) const
{
    bool isolated = true;
    bool keepInstances = true;
    std::vector<const char*> rawMetricNames;

    for (const auto& metricName : metrics)
    {
        std::string parsedName;
        [[maybe_unused]] const bool success = ParseMetricNameString(metricName, parsedName, isolated, keepInstances);
        KttAssert(success, "Unable to parse metric name " + metricName);

        keepInstances = true; // Bug in collection with collection of metrics without instances, keep it to true
        NVPW_MetricEvalRequest evalRequest;

        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params metricToEvalRequest =
        {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE,
            nullptr,
            m_Evaluator,
            parsedName.c_str(),
            &evalRequest,
            NVPW_MetricEvalRequest_STRUCT_SIZE
        };

        CheckError(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&metricToEvalRequest),
            "NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest");
        std::vector<const char*> rawDependencies;

        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params rawDependenciesParams =
        {
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE,
            nullptr,
            m_Evaluator,
            &evalRequest,
            1,
            NVPW_MetricEvalRequest_STRUCT_SIZE,
            sizeof(NVPW_MetricEvalRequest),
            nullptr,
            0,
            nullptr,
            0
        };

        CheckError(NVPW_MetricsEvaluator_GetMetricRawDependencies(&rawDependenciesParams),
            "NVPW_MetricsEvaluator_GetMetricRawDependencies");
        rawDependencies.resize(rawDependenciesParams.numRawDependencies);
        rawDependenciesParams.ppRawDependencies = rawDependencies.data();
        CheckError(NVPW_MetricsEvaluator_GetMetricRawDependencies(&rawDependenciesParams),
            "NVPW_MetricsEvaluator_GetMetricRawDependencies");

        for (size_t i = 0; i < rawDependencies.size(); ++i)
        {
            rawMetricNames.push_back(rawDependencies[i]);
        }
    }

    for (const auto* rawMetricName : rawMetricNames)
    {
        NVPA_RawMetricRequest request =
        {
            NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE,
            nullptr,
            rawMetricName,
            isolated,
            keepInstances
        };

        rawMetricRequests.push_back(request);
    }
}

void CuptiMetricInterface::InitializeCounterAvailabilityImage(const CudaContext& context)
{
    CUpti_Profiler_GetCounterAvailability_Params counterAvailabilityParams =
    {
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
        nullptr,
        context.GetContext(),
        0,
        nullptr
    };

    CheckError(cuptiProfilerGetCounterAvailability(&counterAvailabilityParams), "cuptiProfilerGetCounterAvailability");
    m_CounterAvailabilityImage.resize(counterAvailabilityParams.counterAvailabilityImageSize);
    counterAvailabilityParams.pCounterAvailabilityImage = m_CounterAvailabilityImage.data();
    CheckError(cuptiProfilerGetCounterAvailability(&counterAvailabilityParams), "cuptiProfilerGetCounterAvailability");
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

    keepInstances = false;

    if (outputName.back() == '+')
    {
        keepInstances = true;
        outputName.pop_back();

        if (outputName.empty())
        {
            return false;
        }
    }

    isolated = true;

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

std::string CuptiMetricInterface::GetMetricRollupOpString(const NVPW_RollupOp rollupOp)
{
    switch (rollupOp)
    {
    case NVPW_ROLLUP_OP_AVG:
        return ".avg";
    case NVPW_ROLLUP_OP_MAX:
        return ".max";
    case NVPW_ROLLUP_OP_MIN:
        return ".min";
    case NVPW_ROLLUP_OP_SUM:
        return ".sum";
    default:
        return "";
    }
}

std::string CuptiMetricInterface::GetSubmetricString(const NVPW_Submetric submetric)
{
    switch (submetric)
    {
    case NVPW_SUBMETRIC_NONE:
        return "";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED:
        return ".peak_sustained";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE:
        return ".peak_sustained_active";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED_ACTIVE_PER_SECOND:
        return ".peak_sustained_active.per_second";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED:
        return ".peak_sustained_elapsed";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED_ELAPSED_PER_SECOND:
        return ".peak_sustained_elapsed.per_second";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME:
        return ".peak_sustained_frame";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED_FRAME_PER_SECOND:
        return ".peak_sustained_frame.per_second";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION:
        return ".peak_sustained_region";
    case NVPW_SUBMETRIC_PEAK_SUSTAINED_REGION_PER_SECOND:
        return ".peak_sustained_region.per_second";
    case NVPW_SUBMETRIC_PER_CYCLE_ACTIVE:
        return ".per_cycle_active";
    case NVPW_SUBMETRIC_PER_CYCLE_ELAPSED:
        return ".per_cycle_elapsed";
    case NVPW_SUBMETRIC_PER_CYCLE_IN_FRAME:
        return ".per_cycle_in_frame";
    case NVPW_SUBMETRIC_PER_CYCLE_IN_REGION:
        return ".per_cycle_in_region";
    case NVPW_SUBMETRIC_PER_SECOND:
        return ".per_second";
    case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ACTIVE:
        return ".pct_of_peak_sustained_active";
    case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_ELAPSED:
        return ".pct_of_peak_sustained_elapsed";
    case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_FRAME:
        return ".pct_of_peak_sustained_frame";
    case NVPW_SUBMETRIC_PCT_OF_PEAK_SUSTAINED_REGION:
        return ".pct_of_peak_sustained_region";
    case NVPW_SUBMETRIC_MAX_RATE:
        return ".max_rate";
    case NVPW_SUBMETRIC_PCT:
        return ".pct";
    case NVPW_SUBMETRIC_RATIO:
        return ".ratio";
    default:
        return "";
    }
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
