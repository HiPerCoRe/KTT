#ifdef KTT_PROFILING_CUPTI

#include <sstream>
#include <stdexcept>
#include <cupti_profiler_target.h>
#include <nvperf_target.h>
#include <compute_engine/cuda/cupti/cupti_metric_interface.h>
#include <compute_engine/cuda/cuda_utility.h>
#include <utility/logger.h>

namespace ktt
{

CUPTIMetricInterface::CUPTIMetricInterface(const std::string& deviceName) :
    deviceName(deviceName),
    context(nullptr),
    maxProfiledRanges(2),
    maxRangeNameLength(64)
{
    NVPW_InitializeHost_Params hostParams =
    {
        NVPW_InitializeHost_Params_STRUCT_SIZE,
        nullptr
    };

    checkNVPAError(NVPW_InitializeHost(&hostParams), "NVPW_InitializeHost");

    NVPW_CUDA_MetricsContext_Create_Params params =
    {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
        nullptr,
        deviceName.data()
    };

    checkNVPAError(NVPW_CUDA_MetricsContext_Create(&params), "NVPW_CUDA_MetricsContext_Create");
    context = params.pMetricsContext;
}

CUPTIMetricInterface::~CUPTIMetricInterface()
{
    NVPW_MetricsContext_Destroy_Params params =
    {
        NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
        nullptr,
        context
    };

    checkNVPAError(NVPW_MetricsContext_Destroy(&params), "NVPW_MetricsContext_Destroy");
}

void CUPTIMetricInterface::listSupportedChips() const
{
    NVPW_GetSupportedChipNames_Params params =
    {
        NVPW_GetSupportedChipNames_Params_STRUCT_SIZE,
        nullptr
    };

    checkNVPAError(NVPW_GetSupportedChipNames(&params), "NVPW_GetSupportedChipNames");
    std::stringstream stream;

    for (size_t i = 0; i < params.numChipNames; ++i)
    {
        stream << params.ppChipNames[i];
        if (i + 1 != params.numChipNames)
        {
            stream << ", ";
        }
    }

    Logger::logInfo(std::string("Number of supported chips for CUPTI profiling: ") + std::to_string(params.numChipNames));
    Logger::logInfo(std::string("List of supported chips: ") + stream.str());
}

void CUPTIMetricInterface::listMetrics(const bool listSubMetrics) const
{
    NVPW_MetricsContext_GetMetricNames_Begin_Params params =
    {
        NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE,
        nullptr,
        context
    };

    params.hidePeakSubMetrics = !listSubMetrics;
    params.hidePerCycleSubMetrics = !listSubMetrics;
    params.hidePctOfPeakSubMetrics = !listSubMetrics;
    checkNVPAError(NVPW_MetricsContext_GetMetricNames_Begin(&params), "NVPW_MetricsContext_GetMetricNames_Begin");

    Logger::logInfo(std::string("Total metrics on the chip: ") + std::to_string(params.numMetrics));
    Logger::logInfo("Metrics list:");

    for (size_t i = 0; i < params.numMetrics; ++i)
    {
        Logger::logInfo(params.ppMetricNames[i]);
    }

    NVPW_MetricsContext_GetMetricNames_End_Params endParams =
    {
        NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE,
        nullptr,
        context
    };

    checkNVPAError(NVPW_MetricsContext_GetMetricNames_End(&endParams), "NVPW_MetricsContext_GetMetricNames_End");
}

CUPTIMetricConfiguration CUPTIMetricInterface::createMetricConfiguration(const std::vector<std::string>& metricNames) const
{
    CUPTIMetricConfiguration result;
    result.metricNames = metricNames;
    result.configImage = getConfigImage(metricNames);
    std::vector<uint8_t> prefix = getCounterDataImagePrefix(metricNames);
    createCounterDataImage(prefix, result.counterDataImage, result.scratchBuffer);
    result.maxProfiledRanges = maxProfiledRanges;
    result.dataCollected = false;
    return result;
}

std::vector<CUPTIMetric> CUPTIMetricInterface::getMetricData(const CUPTIMetricConfiguration& configuration) const
{
    if (!configuration.dataCollected)
    {
        throw std::runtime_error("Unable to retrieve metrics from metric configuration with uncollected data");
    }

    const auto& metricNames = configuration.metricNames;
    const auto& counterDataImage = configuration.counterDataImage;

    NVPW_CounterData_GetNumRanges_Params params =
    {
        NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE,
        nullptr,
        counterDataImage.data()
    };

    checkNVPAError(NVPW_CounterData_GetNumRanges(&params), "NVPW_CounterData_GetNumRanges");

    std::vector<CUPTIMetric> result(metricNames.size());
    std::vector<std::string> parsedNames(metricNames.size());
    std::vector<const char*> metricNamePtrs;
    bool isolated = true;
    bool keepInstances = true;

    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
    {
        parseMetricNameString(metricNames[metricIndex], parsedNames[metricIndex], isolated, keepInstances);
        metricNamePtrs.push_back(parsedNames[metricIndex].c_str());
        result[metricIndex].name = metricNames[metricIndex];
        result[metricIndex].rangeCount = params.numRanges;
    }

    for (size_t rangeIndex = 0; rangeIndex < params.numRanges; ++rangeIndex)
    {
        std::vector<const char*> descriptionPtrs;

        NVPW_Profiler_CounterData_GetRangeDescriptions_Params descriptionParams =
        {
            NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE,
            nullptr,
            counterDataImage.data(),
            rangeIndex
        };

        checkNVPAError(NVPW_Profiler_CounterData_GetRangeDescriptions(&descriptionParams), "NVPW_Profiler_CounterData_GetRangeDescriptions");
        descriptionPtrs.resize(descriptionParams.numDescriptions);

        descriptionParams.ppDescriptions = descriptionPtrs.data();
        checkNVPAError(NVPW_Profiler_CounterData_GetRangeDescriptions(&descriptionParams), "NVPW_Profiler_CounterData_GetRangeDescriptions");

        std::string rangeName;

        for (size_t descriptionIndex = 0; descriptionIndex < descriptionParams.numDescriptions; ++descriptionIndex)
        {
            if (descriptionIndex != 0)
            {
                rangeName += "/";
            }

            rangeName += descriptionPtrs[descriptionIndex];
        }

        std::vector<double> gpuValues(metricNames.size());
        NVPW_MetricsContext_SetCounterData_Params dataParams =
        {
            NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE,
            nullptr,
            context,
            counterDataImage.data(),
            rangeIndex,
            isolated
        };

        checkNVPAError(NVPW_MetricsContext_SetCounterData(&dataParams), "NVPW_MetricsContext_SetCounterData");

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalParams =
        {
            NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE,
            nullptr,
            context,
            metricNamePtrs.size(),
            metricNamePtrs.data(),
            gpuValues.data()
        };

        checkNVPAError(NVPW_MetricsContext_EvaluateToGpuValues(&evalParams), "NVPW_MetricsContext_EvaluateToGpuValues");

        for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
        {
            result[metricIndex].rangeToMetricValue.insert(std::make_pair(rangeName, gpuValues[metricIndex]));
        }
    }

    return result;
}

void CUPTIMetricInterface::printMetricValues(const std::vector<CUPTIMetric>& metrics)
{
    for (const auto& metric : metrics)
    {
        Logger::logInfo(std::string("Metric name: ") + metric.name);

        for (const auto& range : metric.rangeToMetricValue)
        {
            Logger::logInfo(std::string("Range name: ") + range.first + "\nGPU value:" + std::to_string(range.second));
        }
    }
}

std::vector<uint8_t> CUPTIMetricInterface::getConfigImage(const std::vector<std::string>& metricNames) const
{
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string> temp;
    getRawMetricRequests(metricNames, rawMetricRequests, temp);

    NVPA_RawMetricsConfigOptions configOptions =
    {
        NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE,
        nullptr,
        NVPA_ACTIVITY_KIND_PROFILER,
        deviceName.c_str()
    };

    NVPA_RawMetricsConfig* rawMetricsConfig;
    checkNVPAError(NVPA_RawMetricsConfig_Create(&configOptions, &rawMetricsConfig), "NVPA_RawMetricsConfig_Create");

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginParams =
    {
        NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig
    };

    checkNVPAError(NVPW_RawMetricsConfig_BeginPassGroup(&beginParams), "NVPW_RawMetricsConfig_BeginPassGroup");

    NVPW_RawMetricsConfig_AddMetrics_Params addParams =
    {
        NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig,
        rawMetricRequests.data(),
        rawMetricRequests.size()
    };

    checkNVPAError(NVPW_RawMetricsConfig_AddMetrics(&addParams), "NVPW_RawMetricsConfig_AddMetrics");

    NVPW_RawMetricsConfig_EndPassGroup_Params endParams =
    {
        NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig
    };

    checkNVPAError(NVPW_RawMetricsConfig_EndPassGroup(&endParams), "NVPW_RawMetricsConfig_EndPassGroup");

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateParams =
    {
        NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig
    };

    checkNVPAError(NVPW_RawMetricsConfig_GenerateConfigImage(&generateParams), "NVPW_RawMetricsConfig_GenerateConfigImage");

    NVPW_RawMetricsConfig_GetConfigImage_Params getParams =
    {
        NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig,
        0,
        nullptr
    };

    checkNVPAError(NVPW_RawMetricsConfig_GetConfigImage(&getParams), "NVPW_RawMetricsConfig_GetConfigImage");

    std::vector<uint8_t> result(getParams.bytesCopied);
    getParams.bytesAllocated = result.size();
    getParams.pBuffer = result.data();

    checkNVPAError(NVPW_RawMetricsConfig_GetConfigImage(&getParams), "NVPW_RawMetricsConfig_GetConfigImage");

    NVPW_RawMetricsConfig_Destroy_Params destroyParams =
    {
        NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
        nullptr,
        rawMetricsConfig
    };

    checkNVPAError(NVPW_RawMetricsConfig_Destroy(&destroyParams), "NVPW_RawMetricsConfig_Destroy");
    return result;
}

std::vector<uint8_t> CUPTIMetricInterface::getCounterDataImagePrefix(const std::vector<std::string>& metricNames) const
{
    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string> temp;
    getRawMetricRequests(metricNames, rawMetricRequests, temp);

    NVPW_CounterDataBuilder_Create_Params createParams =
    {
        NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE,
        nullptr,
        nullptr,
        deviceName.c_str()
    };

    checkNVPAError(NVPW_CounterDataBuilder_Create(&createParams), "NVPW_CounterDataBuilder_Create");

    NVPW_CounterDataBuilder_AddMetrics_Params addParams =
    {
        NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE,
        nullptr,
        createParams.pCounterDataBuilder,
        rawMetricRequests.data(),
        rawMetricRequests.size()
    };

    checkNVPAError(NVPW_CounterDataBuilder_AddMetrics(&addParams), "NVPW_CounterDataBuilder_AddMetrics");

    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getParams =
    {
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
        nullptr,
        createParams.pCounterDataBuilder,
        0,
        nullptr
    };

    checkNVPAError(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getParams), "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    std::vector<uint8_t> result(getParams.bytesCopied);
    getParams.bytesAllocated = result.size();
    getParams.pBuffer = result.data();

    checkNVPAError(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getParams), "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    NVPW_CounterDataBuilder_Destroy_Params destroyParams =
    {
        NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE,
        nullptr,
        createParams.pCounterDataBuilder
    };

    checkNVPAError(NVPW_CounterDataBuilder_Destroy(&destroyParams), "NVPW_CounterDataBuilder_Destroy");
    return result;
}

void CUPTIMetricInterface::createCounterDataImage(const std::vector<uint8_t>& counterDataImagePrefix, std::vector<uint8_t>& counterDataImage,
    std::vector<uint8_t>& counterDataScratchBuffer) const
{
    const CUpti_Profiler_CounterDataImageOptions counterDataImageOptions =
    {
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        nullptr,
        counterDataImagePrefix.data(),
        counterDataImagePrefix.size(),
        maxProfiledRanges,
        maxProfiledRanges,
        maxRangeNameLength
    };

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams =
    {
        CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
        nullptr,
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        &counterDataImageOptions
    };

    checkCUPTIError(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams), "cuptiProfilerCounterDataImageCalculateSize");
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

    checkCUPTIError(cuptiProfilerCounterDataImageInitialize(&initializeParams), "cuptiProfilerCounterDataImageInitialize");

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchSizeParams =
    {
        CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        nullptr,
        counterDataImage.size(),
        counterDataImage.data()
    };

    checkCUPTIError(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchSizeParams),
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

    checkCUPTIError(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initializeScratchParams),
        "cuptiProfilerCounterDataImageInitializeScratchBuffer");
}

void CUPTIMetricInterface::getRawMetricRequests(const std::vector<std::string>& metricNames, std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
    std::vector<std::string>& temp) const
{
    std::string parsedName;
    bool isolated = true;
    bool keepInstances = true;

    for (const auto& metricName : metricNames)
    {
        parseMetricNameString(metricName, parsedName, isolated, keepInstances);
        keepInstances = true; // Bug in collection with collection of metrics without instances, keep it to true

        NVPW_MetricsContext_GetMetricProperties_Begin_Params params =
        {
            NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
            nullptr,
            context,
            parsedName.c_str()
        };

        checkNVPAError(NVPW_MetricsContext_GetMetricProperties_Begin(&params), "NVPW_MetricsContext_GetMetricProperties_Begin");

        for (const char** metricDependencies = params.ppRawMetricDependencies; *metricDependencies != nullptr; ++metricDependencies)
        {
            temp.push_back(*metricDependencies);
        }

        NVPW_MetricsContext_GetMetricProperties_End_Params endParams =
        {
            NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
            nullptr,
            context
        };

        checkNVPAError(NVPW_MetricsContext_GetMetricProperties_End(&endParams), "NVPW_MetricsContext_GetMetricProperties_End");
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

void CUPTIMetricInterface::checkNVPAError(const NVPA_Status value, const std::string& message)
{
    if (value != NVPA_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string("CUPTI error: ") + std::to_string(static_cast<int>(value)) + "\nAdditional info: " + message);
    }
}

bool CUPTIMetricInterface::parseMetricNameString(const std::string& metricName, std::string& outputName, bool& isolated, bool& keepInstances)
{
    if (metricName.empty())
    {
        outputName = "";
        return false;
    }

    outputName = metricName;
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
