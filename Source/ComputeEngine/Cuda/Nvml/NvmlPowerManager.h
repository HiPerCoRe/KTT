#pragma once

#if defined(KTT_POWER_USAGE_NVML)

#include <atomic>
#include <cstdint>
#include <vector>
#include <ctpl_stl.h>
#include <nvml.h>

#include <KttTypes.h>

namespace ktt
{

class CudaContext;

class NvmlPowerManager
{
public:
    // Note that sampling frequency over 300 will cause large variance in number of collected samples due to driver overhead
    explicit NvmlPowerManager(const CudaContext& context, const DeviceIndex device, const uint32_t samplingFrequency = 100);
    ~NvmlPowerManager();

    void StartCollection();
    void EndCollection();
    uint32_t GetPowerUsage() const;
    uint64_t GetTotalDeviceEnergy() const;
    double GetTemperature() const;
    uint32_t GetSMFrequency() const;
    uint32_t GetMemoryFrequency() const;

private:
    const CudaContext& m_Context;
    nvmlDevice_t m_Device;
    ctpl::thread_pool m_Pool;
    std::future<void> m_Future;
    std::atomic<bool> m_StopFlag;
    std::vector<uint32_t> m_PowerSamples;
    std::vector<uint32_t> m_TempSamples;
    std::vector<uint32_t> m_SMFreqSamples;
    std::vector<uint32_t> m_MemFreqSamples;
    Nanoseconds m_SamplingInterval;

    void CollectPowerSamples();
};

} // namespace ktt

#endif // KTT_POWER_USAGE_NVML
