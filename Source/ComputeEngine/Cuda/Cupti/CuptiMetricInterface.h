#pragma once

#ifdef KTT_PROFILING_CUPTI

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <nvperf_host.h>

#include <ComputeEngine/Cuda/Cupti/CuptiMetricConfiguration.h>
#include <KttTypes.h>

namespace ktt
{

class CudaContext;
class KernelProfilingData;

class CuptiMetricInterface
{
public:
    CuptiMetricInterface(const DeviceIndex index, const CudaContext& context);
    ~CuptiMetricInterface();

    void SetMetrics(const std::vector<std::string>& metrics);

    CuptiMetricConfiguration CreateMetricConfiguration() const;
    std::unique_ptr<KernelProfilingData> GenerateProfilingData(const CuptiMetricConfiguration& configuration) const;

    static void ListSupportedChips();

private:
    std::vector<std::string> m_Metrics;
    std::vector<uint8_t> m_CounterAvailabilityImage;
    std::vector<uint8_t> m_ScratchBuffer;
    std::string m_DeviceName;
    NVPW_MetricsEvaluator* m_Evaluator;
    uint32_t m_MaxProfiledRanges;
    uint32_t m_MaxRangeNameLength;

    std::set<std::string> GetSupportedMetrics(const bool listSubMetrics) const;
    std::vector<uint8_t> GetConfigImage(const std::vector<std::string>& metrics) const;
    std::vector<uint8_t> GetCounterDataImagePrefix(const std::vector<std::string>& metrics) const;
    void CreateCounterDataImage(const std::vector<uint8_t>& counterDataImagePrefix, std::vector<uint8_t>& counterDataImage,
        std::vector<uint8_t>& counterDataScratchBuffer) const;
    void GetRawMetricRequests(const std::vector<std::string>& metrics, std::vector<NVPA_RawMetricRequest>& rawMetricRequests) const;
    void InitializeCounterAvailabilityImage(const CudaContext& context);

    static const std::vector<std::string>& GetDefaultMetrics();
    static std::string GetDeviceName(const DeviceIndex index);
    static bool ParseMetricNameString(const std::string& metric, std::string& outputName, bool& isolated, bool& keepInstances);
    static std::string GetMetricRollupOpString(const NVPW_RollupOp rollupOp);
    static std::string GetSubmetricString(const NVPW_Submetric submetric);
};

} // namespace ktt

#endif // KTT_PROFILING_CUPTI
