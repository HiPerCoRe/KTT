#pragma once

#ifdef KTT_PROFILING_CUPTI

#include <cstddef>
#include <vector>

namespace ktt
{

struct MetricNameValue
{
    std::string metricName;
    size_t numRanges;
    // <rangeName , metricValue> pair
    std::vector<std::pair<std::string, double>> rangeNameMetricValueMap;
};

void checkNVPAError(const NVPA_Status value, const std::string& message);
std::string getHwUnit(const std::string& metricName);
void listSupportedChips();
void listMetrics(const char* chip, const bool listSubMetrics);

void getMetricGpuValue(const std::string& chipName, const std::vector<uint8_t>& counterDataImage, const std::vector<std::string>& metricNames,
    std::vector<MetricNameValue>& metricNameValueMap);
void printMetricValues(const std::string& chipName, const std::vector<uint8_t>& counterDataImage, const std::vector<std::string>& metricNames);
void getRawMetricRequests(NVPA_MetricsContext* pMetricsContext, const std::vector<std::string>& metricNames,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests, std::vector<std::string>& temp);
void getConfigImage(const std::string& chipName, const std::vector<std::string>& metricNames, std::vector<uint8_t>& configImage);
void getCounterDataPrefixImage(const std::string& chipName, const std::vector<std::string>& metricNames, std::vector<uint8_t>& counterDataImagePrefix);

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
