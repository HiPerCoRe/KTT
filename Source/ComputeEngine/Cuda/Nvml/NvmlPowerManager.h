#pragma once

#if defined(KTT_POWER_USAGE_NVML)

#include <atomic>
#include <chrono>
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
    // Restricts the samples used by the Get*() averaging methods to those collected while the
    // kernel was actually executing on the GPU. The execution window is anchored at the moment
    // sampling stopped (kernel completion, captured in EndCollection) and extends backwards by
    // the precise kernel duration measured via CUDA events. This excludes idle-power samples
    // taken before the kernel was submitted and after it finished, which would otherwise dilute
    // the averages. Must be called after EndCollection and before the Get*() methods.
    void RestrictToExecutionWindow(Nanoseconds kernelDuration);
    uint32_t GetPowerUsage() const;
    uint64_t GetTotalDeviceEnergy() const;
    double GetTemperature() const;
    uint32_t GetSMFrequency() const;
    uint32_t GetMemoryFrequency() const;
    int32_t GetFanSpeed() const;

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
    std::vector<int32_t> m_FanSpeedSamples;
    std::vector<std::chrono::steady_clock::time_point> m_SampleTimes;
    std::chrono::steady_clock::time_point m_CollectionStopTime;
    size_t m_WindowBegin;
    size_t m_WindowEnd;
    Nanoseconds m_SamplingInterval;

    void CollectPowerSamples();
    size_t FindNearestSample(std::chrono::steady_clock::time_point windowStart,
        std::chrono::steady_clock::time_point windowEnd) const;
};

} // namespace ktt

#endif // KTT_POWER_USAGE_NVML
