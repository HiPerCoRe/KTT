#pragma once

#ifdef KTT_PROFILING_CUPTI

#include <cstdint>
#include <string>
#include <vector>
#include <nvperf_host.h>

#include <ComputeEngine/Cuda/Cupti/CuptiMetric.h>
#include <ComputeEngine/Cuda/Cupti/CuptiMetricConfiguration.h>
#include <KttTypes.h>

namespace ktt
{

class CuptiMetricInterface
{
public:
    CuptiMetricInterface(const DeviceIndex index);
    ~CuptiMetricInterface();

    static void ListSupportedChips();
    void ListMetrics(const bool listSubMetrics) const;

    CuptiMetricConfiguration CreateMetricConfiguration(const std::vector<std::string>& metrics) const;
    std::vector<CuptiMetric> GenerateMetrics(const CuptiMetricConfiguration& configuration) const;

private:
    std::string m_DeviceName;
    NVPA_MetricsContext* m_Context;
    uint32_t m_MaxProfiledRanges;
    uint32_t m_MaxRangeNameLength;

    std::vector<uint8_t> GetConfigImage(const std::vector<std::string>& metrics) const;
    std::vector<uint8_t> GetCounterDataImagePrefix(const std::vector<std::string>& metrics) const;
    void CreateCounterDataImage(const std::vector<uint8_t>& counterDataImagePrefix, std::vector<uint8_t>& counterDataImage,
        std::vector<uint8_t>& counterDataScratchBuffer) const;
    void GetRawMetricRequests(const std::vector<std::string>& metrics, std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
        std::vector<std::string>& temp) const;
    static std::string GetDeviceName(const DeviceIndex index);
    static bool ParseMetricNameString(const std::string& metric, std::string& outputName, bool& isolated, bool& keepInstances);
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
