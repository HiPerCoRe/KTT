#ifdef KTT_PROFILING_CUPTI

#include <stdexcept>
#include <string>
#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>
#include <compute_engine/cuda/cupti/cupti_metric_utility.h>
#include <utility/logger.h>

namespace ktt
{

void checkNVPAError(const NVPA_Status value, const std::string& message)
{
    if (value != NVPA_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string("Internal CUPTI error: ") + std::to_string(static_cast<int>(value)) + "\nAdditional info: " + message);
    }
}

std::string getHwUnit(const std::string& metricName)
{
    return metricName.substr(0, metricName.find("__"));
}

void listSupportedChips()
{
    NVPW_GetSupportedChipNames_Params getSupportedChipNames = { NVPW_GetSupportedChipNames_Params_STRUCT_SIZE };
    checkNVPAError(NVPW_GetSupportedChipNames(&getSupportedChipNames), "NVPW_GetSupportedChipNames");
    Logger::logInfo(std::string("Number of supported chips: ") + std::to_string(getSupportedChipNames.numChipNames));
    Logger::logInfo("List of supported chips:");

    for (size_t i = 0; i < getSupportedChipNames.numChipNames; ++i)
    {
        Logger::logInfo(getSupportedChipNames.ppChipNames[i]);
    }
}

void listMetrics(const char* chip, const bool listSubMetrics)
{
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chip;
    checkNVPAError(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams), "NVPW_CUDA_MetricsContext_Create");

    NVPW_MetricsContext_GetMetricNames_Begin_Params getMetricNameBeginParams = { NVPW_MetricsContext_GetMetricNames_Begin_Params_STRUCT_SIZE };
    getMetricNameBeginParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    getMetricNameBeginParams.hidePeakSubMetrics = !listSubMetrics;
    getMetricNameBeginParams.hidePerCycleSubMetrics = !listSubMetrics;
    getMetricNameBeginParams.hidePctOfPeakSubMetrics = !listSubMetrics;
    checkNVPAError(NVPW_MetricsContext_GetMetricNames_Begin(&getMetricNameBeginParams), "NVPW_MetricsContext_GetMetricNames_Begin");

    Logger::logInfo(std::string("Total metrics on the chip: ") + std::to_string(getMetricNameBeginParams.numMetrics));
    Logger::logInfo("Metrics list:");

    for (size_t i = 0; i < getMetricNameBeginParams.numMetrics; ++i)
    {
        Logger::logInfo(getMetricNameBeginParams.ppMetricNames[i]);
    }

    NVPW_MetricsContext_GetMetricNames_End_Params getMetricNameEndParams = { NVPW_MetricsContext_GetMetricNames_End_Params_STRUCT_SIZE };
    getMetricNameEndParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    NVPW_MetricsContext_GetMetricNames_End(&getMetricNameEndParams);

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    NVPW_MetricsContext_Destroy(&metricsContextDestroyParams);
}

inline bool parseMetricNameString(const std::string& metricName, std::string& reqName, bool& isolated, bool& keepInstances)
{
    std::string& name = reqName;
    name = metricName;

    if (name.empty())
    {
        return false;
    }

    // boost program_options sometimes inserts a \n between the metric name and a '&' at the end
    size_t pos = name.find('\n');
    if (pos != std::string::npos)
    {
        name.erase(pos, 1);
    }

    // trim whitespace
    while (name.back() == ' ')
    {
        name.pop_back();
        if (name.empty())
        {
            return false;
        }
    }

    keepInstances = false;
    if (name.back() == '+')
    {
        keepInstances = true;
        name.pop_back();
        if (name.empty())
        {
            return false;
        }
    }

    isolated = true;
    if (name.back() == '$')
    {
        name.pop_back();
        if (name.empty())
        {
            return false;
        }
    }
    else if (name.back() == '&')
    {
        isolated = false;
        name.pop_back();
        if (name.empty())
        {
            return false;
        }
    }

    return true;
}

void getMetricGpuValue(const std::string& chipName, const std::vector<uint8_t>& counterDataImage, const std::vector<std::string>& metricNames,
    std::vector<MetricNameValue>& metricNameValueMap)
{
    if (counterDataImage.empty())
    {
        throw std::runtime_error("Counter Data Image is empty");
    }

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chipName.c_str();
    checkNVPAError(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams), "NVPW_CUDA_MetricsContext_Create");

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
    getNumRangesParams.pCounterDataImage = counterDataImage.data();
    checkNVPAError(NVPW_CounterData_GetNumRanges(&getNumRangesParams), "NVPW_CounterData_GetNumRanges");

    std::vector<std::string> reqName(metricNames.size());
    bool isolated = true;
    bool keepInstances = true;
    std::vector<const char*> metricNamePtrs;
    metricNameValueMap.resize(metricNames.size());

    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
    {
        parseMetricNameString(metricNames[metricIndex], reqName[metricIndex], isolated, keepInstances);
        metricNamePtrs.push_back(reqName[metricIndex].c_str());
        metricNameValueMap[metricIndex].metricName = metricNames[metricIndex];
        metricNameValueMap[metricIndex].numRanges = getNumRangesParams.numRanges;
    }

    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
    {
        std::vector<const char*> descriptionPtrs;

        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
        getRangeDescParams.pCounterDataImage = counterDataImage.data();
        getRangeDescParams.rangeIndex = rangeIndex;
        checkNVPAError(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams), "NVPW_Profiler_CounterData_GetRangeDescriptions");
        descriptionPtrs.resize(getRangeDescParams.numDescriptions);

        getRangeDescParams.ppDescriptions = descriptionPtrs.data();
        checkNVPAError(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams), "NVPW_Profiler_CounterData_GetRangeDescriptions");

        std::string rangeName;

        for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
        {
            if (descriptionIndex != 0)
            {
                rangeName += "/";
            }

            rangeName += descriptionPtrs[descriptionIndex];
        }

        std::vector<double> gpuValues(metricNames.size());
        NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
        setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        setCounterDataParams.pCounterDataImage = counterDataImage.data();
        setCounterDataParams.isolated = true;
        setCounterDataParams.rangeIndex = rangeIndex;
        checkNVPAError(NVPW_MetricsContext_SetCounterData(&setCounterDataParams), "NVPW_MetricsContext_SetCounterData");

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
        evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        evalToGpuParams.numMetrics = metricNamePtrs.size();
        evalToGpuParams.ppMetricNames = metricNamePtrs.data();
        evalToGpuParams.pMetricValues = gpuValues.data();
        checkNVPAError(NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams), "NVPW_MetricsContext_EvaluateToGpuValues");

        for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
        {
            metricNameValueMap[metricIndex].rangeNameMetricValueMap.push_back(std::make_pair(rangeName, gpuValues[metricIndex]));
        }
    }

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    NVPW_MetricsContext_Destroy(&metricsContextDestroyParams);
}

void printMetricValues(const std::string& chipName, const std::vector<uint8_t>& counterDataImage, const std::vector<std::string>& metricNames)
{
    if (counterDataImage.empty())
    {
        throw std::runtime_error("Counter Data Image is empty");
    }

    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chipName.c_str();
    checkNVPAError(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams), "NVPW_CUDA_MetricsContext_Create");

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
    getNumRangesParams.pCounterDataImage = counterDataImage.data();
    checkNVPAError(NVPW_CounterData_GetNumRanges(&getNumRangesParams), "NVPW_CounterData_GetNumRanges");

    std::vector<std::string> reqName(metricNames.size());
    bool isolated = true;
    bool keepInstances = true;
    std::vector<const char*> metricNamePtrs;

    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
    {
        parseMetricNameString(metricNames[metricIndex], reqName[metricIndex], isolated, keepInstances);
        metricNamePtrs.push_back(reqName[metricIndex].c_str());
    }

    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
    {
        std::vector<const char*> descriptionPtrs;

        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
        getRangeDescParams.pCounterDataImage = counterDataImage.data();
        getRangeDescParams.rangeIndex = rangeIndex;
        checkNVPAError(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams), "NVPW_Profiler_CounterData_GetRangeDescriptions");

        descriptionPtrs.resize(getRangeDescParams.numDescriptions);

        getRangeDescParams.ppDescriptions = &descriptionPtrs[0];
        checkNVPAError(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams), "NVPW_Profiler_CounterData_GetRangeDescriptions");

        std::string rangeName;
        for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
        {
            if (descriptionIndex != 0)
            {
                rangeName += "/";
            }
            rangeName += descriptionPtrs[descriptionIndex];
        }

        const bool isolated = true;
        std::vector<double> gpuValues;
        gpuValues.resize(metricNames.size());

        NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = { NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE };
        setCounterDataParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        setCounterDataParams.pCounterDataImage = counterDataImage.data();
        setCounterDataParams.isolated = true;
        setCounterDataParams.rangeIndex = rangeIndex;
        NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = { NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE };
        evalToGpuParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
        evalToGpuParams.numMetrics = metricNamePtrs.size();
        evalToGpuParams.ppMetricNames = &metricNamePtrs[0];
        evalToGpuParams.pMetricValues = &gpuValues[0];
        NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);

        for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
        {
            Logger::logInfo(std::string("Range name: ") + rangeName + "\nMetric name:" + metricNames[metricIndex] + "\n GPU value:"
                + std::to_string(gpuValues[metricIndex]));
        }
    }
    
    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    NVPW_MetricsContext_Destroy(&metricsContextDestroyParams);
}

void getRawMetricRequests(NVPA_MetricsContext* pMetricsContext, const std::vector<std::string>& metricNames,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests, std::vector<std::string>& temp)
{
    std::string reqName;
    bool isolated = true;
    bool keepInstances = true;

    for (const auto& metricName : metricNames)
    {
        parseMetricNameString(metricName, reqName, isolated, keepInstances);
        /* Bug in collection with collection of metrics without instances, keep it to true*/
        keepInstances = true;
        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = { NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE };
        getMetricPropertiesBeginParams.pMetricsContext = pMetricsContext;
        getMetricPropertiesBeginParams.pMetricName = reqName.c_str();

        checkNVPAError(NVPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams), "NVPW_MetricsContext_GetMetricProperties_Begin");

        for (const char** ppMetricDependencies = getMetricPropertiesBeginParams.ppRawMetricDependencies; *ppMetricDependencies; ++ppMetricDependencies)
        {
            temp.push_back(*ppMetricDependencies);
        }

        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = { NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE };
        getMetricPropertiesEndParams.pMetricsContext = pMetricsContext;
        checkNVPAError(NVPW_MetricsContext_GetMetricProperties_End(&getMetricPropertiesEndParams), "NVPW_MetricsContext_GetMetricProperties_End");
    }

    for (const auto& rawMetricName : temp)
    {
        NVPA_RawMetricRequest metricRequest = { NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE };
        metricRequest.pMetricName = rawMetricName.c_str();
        metricRequest.isolated = isolated;
        metricRequest.keepInstances = keepInstances;
        rawMetricRequests.push_back(metricRequest);
    }
}

void getConfigImage(const std::string& chipName, const std::vector<std::string>& metricNames, std::vector<uint8_t>& configImage)
{
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chipName.c_str();
    checkNVPAError(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams), "NVPW_CUDA_MetricsContext_Create");

    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string> temp;
    getRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);

    NVPA_RawMetricsConfigOptions metricsConfigOptions = { NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE };
    metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    metricsConfigOptions.pChipName = chipName.c_str();

    NVPA_RawMetricsConfig* pRawMetricsConfig;
    checkNVPAError(NVPA_RawMetricsConfig_Create(&metricsConfigOptions, &pRawMetricsConfig), "NVPA_RawMetricsConfig_Create");

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = { NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE };
    beginPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    checkNVPAError(NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams), "NVPW_RawMetricsConfig_BeginPassGroup");

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = { NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pRawMetricsConfig = pRawMetricsConfig;
    addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
    addMetricsParams.numMetricRequests = rawMetricRequests.size();
    checkNVPAError(NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams), "NVPW_RawMetricsConfig_AddMetrics");

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = { NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE };
    endPassGroupParams.pRawMetricsConfig = pRawMetricsConfig;
    checkNVPAError(NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams), "NVPW_RawMetricsConfig_EndPassGroup");

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = { NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE };
    generateConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    checkNVPAError(NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams), "NVPW_RawMetricsConfig_GenerateConfigImage");

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = { NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE };
    getConfigImageParams.pRawMetricsConfig = pRawMetricsConfig;
    getConfigImageParams.bytesAllocated = 0;
    getConfigImageParams.pBuffer = NULL;
    checkNVPAError(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams), "NVPW_RawMetricsConfig_GetConfigImage");

    configImage.resize(getConfigImageParams.bytesCopied);

    getConfigImageParams.bytesAllocated = configImage.size();
    getConfigImageParams.pBuffer = configImage.data();
    checkNVPAError(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams), "NVPW_RawMetricsConfig_GetConfigImage");

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = { NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE };
    rawMetricsConfigDestroyParams.pRawMetricsConfig = pRawMetricsConfig;
    NVPW_RawMetricsConfig_Destroy(&rawMetricsConfigDestroyParams);

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    NVPW_MetricsContext_Destroy(&metricsContextDestroyParams);
}

void getCounterDataPrefixImage(const std::string& chipName, const std::vector<std::string>& metricNames, std::vector<uint8_t>& counterDataImagePrefix)
{
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = { NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE };
    metricsContextCreateParams.pChipName = chipName.c_str();
    checkNVPAError(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams), "NVPW_CUDA_MetricsContext_Create");

    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<std::string> temp;
    getRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);

    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = { NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE };
    counterDataBuilderCreateParams.pChipName = chipName.c_str();
    checkNVPAError(NVPW_CounterDataBuilder_Create(&counterDataBuilderCreateParams), "NVPW_CounterDataBuilder_Create");

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = { NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE };
    addMetricsParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    addMetricsParams.pRawMetricRequests = &rawMetricRequests[0];
    addMetricsParams.numMetricRequests = rawMetricRequests.size();
    checkNVPAError(NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams), "NVPW_CounterDataBuilder_AddMetrics");

    size_t counterDataPrefixSize = 0;
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = { NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE };
    getCounterDataPrefixParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    getCounterDataPrefixParams.bytesAllocated = 0;
    getCounterDataPrefixParams.pBuffer = NULL;
    checkNVPAError(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams), "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);

    getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
    getCounterDataPrefixParams.pBuffer = counterDataImagePrefix.data();
    checkNVPAError(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams), "NVPW_CounterDataBuilder_GetCounterDataPrefix");

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = { NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE };
    counterDataBuilderDestroyParams.pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder;
    NVPW_CounterDataBuilder_Destroy(&counterDataBuilderDestroyParams);

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = { NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE };
    metricsContextDestroyParams.pMetricsContext = metricsContextCreateParams.pMetricsContext;
    NVPW_MetricsContext_Destroy(&metricsContextDestroyParams);
}

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
