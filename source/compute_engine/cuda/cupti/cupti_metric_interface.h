#pragma once

#ifdef KTT_PROFILING_CUPTI

#include <cstddef>
#include <string>
#include <vector>
#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <compute_engine/cuda/cupti/cupti_metric.h>

namespace ktt
{

class CUPTIMetricInterface
{
public:
    CUPTIMetricInterface(const std::string& deviceName);
    ~CUPTIMetricInterface();

    void listSupportedChips() const;
    void listMetrics(const bool listSubMetrics) const;

    std::vector<uint8_t> getConfigImage(const std::vector<std::string>& metricNames) const;
    std::vector<uint8_t> getCounterDataPrefixImage(const std::vector<std::string>& metricNames) const;
    std::vector<CUPTIMetric> getMetricData(const std::vector<std::string>& metricNames, const std::vector<uint8_t>& counterDataImage) const;
    static void printMetricValues(const std::vector<CUPTIMetric>& metrics);

private:
    std::string deviceName;
    NVPA_MetricsContext* context;

    void getRawMetricRequests(const std::vector<std::string>& metricNames, std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
        std::vector<std::string>& temp) const;
    static void checkNVPAError(const NVPA_Status value, const std::string& message);
    static bool parseMetricNameString(const std::string& metricName, std::string& outputName, bool& isolated, bool& keepInstances);
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
