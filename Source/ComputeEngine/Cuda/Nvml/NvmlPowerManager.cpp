#if defined(KTT_POWER_USAGE_NVML)

#include <algorithm>
#include <numeric>

#include <ComputeEngine/Cuda/Nvml/NvmlPowerManager.h>
#include <ComputeEngine/Cuda/CudaContext.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/Timer.h>

namespace ktt
{

NvmlPowerManager::NvmlPowerManager(const CudaContext& context, const DeviceIndex device, const uint32_t samplingFrequency) :
    m_Context(context),
    m_Pool(1),
    m_StopFlag(true),
    m_WindowBegin(0),
    m_WindowEnd(0),
    m_SamplingInterval(static_cast<Nanoseconds>(1.0 / static_cast<double>(samplingFrequency) * 1'000'000'000.0))
{
    CheckError(nvmlInit_v2(), "nvmlInit_v2");
    CheckError(nvmlDeviceGetHandleByIndex(device, &m_Device), "nvmlDeviceGetHandleByIndex");
    m_PowerSamples.reserve(samplingFrequency);
    m_TempSamples.reserve(samplingFrequency);
    m_SMFreqSamples.reserve(samplingFrequency);
    m_MemFreqSamples.reserve(samplingFrequency);
    m_FanSpeedSamples.reserve(samplingFrequency);
    m_SampleTimes.reserve(samplingFrequency);
}

NvmlPowerManager::~NvmlPowerManager()
{
    CheckError(nvmlShutdown(), "nvmlShutdown");
}

void NvmlPowerManager::StartCollection()
{
    m_Context.Synchronize();

    m_PowerSamples.clear();
    m_TempSamples.clear();
    m_SMFreqSamples.clear();
    m_MemFreqSamples.clear();
    m_FanSpeedSamples.clear();
    m_SampleTimes.clear();
    // Until RestrictToExecutionWindow is called, the averaging methods consider every sample.
    m_WindowBegin = 0;
    m_WindowEnd = 0;
    m_StopFlag = false;

    m_Future = m_Pool.push([this]()
    {
        CollectPowerSamples();
    });
}

void NvmlPowerManager::EndCollection()
{
    m_Context.Synchronize();

    // The caller invokes EndCollection right after waiting for the kernel to finish, so this is
    // the closest host-side proxy for the moment GPU execution ended. RestrictToExecutionWindow
    // uses it as the anchor for the execution window.
    m_CollectionStopTime = std::chrono::steady_clock::now();
    m_StopFlag = true;
    m_Future.wait();

    // By default consider every collected sample; RestrictToExecutionWindow narrows this.
    m_WindowBegin = 0;
    m_WindowEnd = m_PowerSamples.size();
}

void NvmlPowerManager::RestrictToExecutionWindow(const Nanoseconds kernelDuration)
{
    m_WindowBegin = 0;
    m_WindowEnd = m_SampleTimes.size();

    if (m_SampleTimes.empty())
    {
        return;
    }

    const auto windowEnd = m_CollectionStopTime;
    const auto windowStart = windowEnd - std::chrono::nanoseconds(kernelDuration);

    // Samples are recorded in chronological order, so the in-window samples form a contiguous range.
    const auto beginIt = std::lower_bound(m_SampleTimes.cbegin(), m_SampleTimes.cend(), windowStart);
    const auto endIt = std::upper_bound(m_SampleTimes.cbegin(), m_SampleTimes.cend(), windowEnd);

    size_t begin = static_cast<size_t>(beginIt - m_SampleTimes.cbegin());
    size_t end = static_cast<size_t>(endIt - m_SampleTimes.cbegin());

    if (begin >= end)
    {
        // The kernel was shorter than the sampling interval, so no sample landed inside the
        // execution window. Fall back to the single sample closest to the window - it is the
        // best available estimate of the in-kernel reading.
        const size_t nearest = FindNearestSample(windowStart, windowEnd);
        begin = nearest;
        end = nearest + 1;
    }

    m_WindowBegin = begin;
    m_WindowEnd = end;
    Logger::LogDebug("Power samples restricted to execution window: using " + std::to_string(end - begin)
        + " of " + std::to_string(m_SampleTimes.size()) + " samples");
}

size_t NvmlPowerManager::FindNearestSample(const std::chrono::steady_clock::time_point windowStart,
    const std::chrono::steady_clock::time_point windowEnd) const
{
    size_t best = 0;
    auto bestDistance = std::chrono::steady_clock::duration::max();

    for (size_t i = 0; i < m_SampleTimes.size(); ++i)
    {
        const auto time = m_SampleTimes[i];
        std::chrono::steady_clock::duration distance;

        if (time < windowStart)
        {
            distance = windowStart - time;
        }
        else if (time > windowEnd)
        {
            distance = time - windowEnd;
        }
        else
        {
            distance = std::chrono::steady_clock::duration::zero();
        }

        if (distance < bestDistance)
        {
            bestDistance = distance;
            best = i;
        }
    }

    return best;
}

uint32_t NvmlPowerManager::GetPowerUsage() const
{
    const size_t count = m_WindowEnd - m_WindowBegin;
    Logger::LogDebug("Generating average power usage from number of samples: " + std::to_string(count));

    if (count == 0)
    {
        return 0;
    }

    const uint32_t sum = std::accumulate(m_PowerSamples.cbegin() + m_WindowBegin, m_PowerSamples.cbegin() + m_WindowEnd, 0);
    return sum / static_cast<uint32_t>(count);
}

uint64_t NvmlPowerManager::GetTotalDeviceEnergy() const
{
    m_Context.Synchronize();
    long long unsigned int energy;
    CheckError(nvmlDeviceGetTotalEnergyConsumption(m_Device, &energy), "nvmlDeviceGetTotalEnergyConsumption");
    return energy;
}

double NvmlPowerManager::GetTemperature() const
{
    const size_t count = m_WindowEnd - m_WindowBegin;
    Logger::LogDebug("Generating average temperatures from number of samples: " + std::to_string(count));

    if (count == 0)
    {
        return 0;
    }

    const double sum = std::accumulate(m_TempSamples.cbegin() + m_WindowBegin, m_TempSamples.cbegin() + m_WindowEnd, 0);
    return sum / static_cast<double>(count);
}

uint32_t NvmlPowerManager::GetSMFrequency() const
{
    const size_t count = m_WindowEnd - m_WindowBegin;
    Logger::LogDebug("Generating average SM frequency from number of samples: " + std::to_string(count));

    if (count == 0)
    {
        return 0;
    }

    const uint32_t sum = std::accumulate(m_SMFreqSamples.cbegin() + m_WindowBegin, m_SMFreqSamples.cbegin() + m_WindowEnd, 0);
    return sum / static_cast<uint32_t>(count);
}

uint32_t NvmlPowerManager::GetMemoryFrequency() const
{
    const size_t count = m_WindowEnd - m_WindowBegin;
    Logger::LogDebug("Generating average memory frequency from number of samples: " + std::to_string(count));

    if (count == 0)
    {
        return 0;
    }

    const uint32_t sum = std::accumulate(m_MemFreqSamples.cbegin() + m_WindowBegin, m_MemFreqSamples.cbegin() + m_WindowEnd, 0);
    return sum / static_cast<uint32_t>(count);
}

int32_t NvmlPowerManager::GetFanSpeed() const
{
    const size_t count = m_WindowEnd - m_WindowBegin;
    Logger::LogDebug("Generating average fan speed from number of samples: " + std::to_string(count));

    if (count == 0)
    {
        return -1;
    }

    // If first sample is -1, fan speed is not supported by the device
    if (m_FanSpeedSamples[m_WindowBegin] == -1)
    {
        return -1;
    }

    // Calculate average of the in-window samples
    const int64_t sum = std::accumulate(m_FanSpeedSamples.cbegin() + m_WindowBegin, m_FanSpeedSamples.cbegin() + m_WindowEnd,
        static_cast<int64_t>(0));
    return static_cast<int32_t>(sum / static_cast<int64_t>(count));
}

void NvmlPowerManager::CollectPowerSamples()
{
    m_SampleTimes.push_back(std::chrono::steady_clock::now());
    uint32_t initialValue;
    CheckError(nvmlDeviceGetPowerUsage(m_Device, &initialValue), "nvmlDeviceGetPowerUsage");
    m_PowerSamples.push_back(initialValue);
    CheckError(nvmlDeviceGetTemperature(m_Device, NVML_TEMPERATURE_GPU, &initialValue), "nvmlDeviceGetTemperature");
    m_TempSamples.push_back(initialValue);
    CheckError(nvmlDeviceGetClockInfo(m_Device, NVML_CLOCK_SM, &initialValue), "nvmlDeviceGetClockInfo");
    m_SMFreqSamples.push_back(initialValue);
    CheckError(nvmlDeviceGetClockInfo(m_Device, NVML_CLOCK_MEM, &initialValue), "nvmlDeviceGetClockInfo");
    m_MemFreqSamples.push_back(initialValue);

    // Get fan speed, handle not supported case
    uint32_t fanSpeed;
    nvmlReturn_t fanResult = nvmlDeviceGetFanSpeed(m_Device, &fanSpeed);
    if (fanResult == NVML_ERROR_NOT_SUPPORTED)
    {
        m_FanSpeedSamples.push_back(-1);
    }
    else
    {
        CheckError(fanResult, "nvmlDeviceGetFanSpeed");
        m_FanSpeedSamples.push_back(static_cast<int32_t>(fanSpeed));
    }

    Timer timer;
    timer.Start();
    
    while (!m_StopFlag.load())
    {
        if (timer.GetCheckpointTime() < m_SamplingInterval)
        {
            continue;
        }

        timer.Restart();

        m_SampleTimes.push_back(std::chrono::steady_clock::now());
        uint32_t value;
        CheckError(nvmlDeviceGetPowerUsage(m_Device, &value), "nvmlDeviceGetPowerUsage");
        m_PowerSamples.push_back(value);

	    uint32_t temp;
	    CheckError(nvmlDeviceGetTemperature(m_Device, NVML_TEMPERATURE_GPU, &temp), "nvmlDeviceGetTemperature");
	    m_TempSamples.push_back(temp);

	    uint32_t smClk, memClk;
	    CheckError(nvmlDeviceGetClockInfo(m_Device, NVML_CLOCK_SM, &smClk), "nvmlDeviceGetClockInfo");
	    CheckError(nvmlDeviceGetClockInfo(m_Device, NVML_CLOCK_MEM, &memClk), "nvmlDeviceGetClockInfo");
	    m_SMFreqSamples.push_back(smClk);
	    m_MemFreqSamples.push_back(memClk);

	    // Get fan speed, handle not supported case
	    uint32_t fanSpeed;
	    nvmlReturn_t fanResult = nvmlDeviceGetFanSpeed(m_Device, &fanSpeed);
	    if (fanResult == NVML_ERROR_NOT_SUPPORTED)
	    {
	        m_FanSpeedSamples.push_back(-1);
	    }
	    else
	    {
	        CheckError(fanResult, "nvmlDeviceGetFanSpeed");
	        m_FanSpeedSamples.push_back(static_cast<int32_t>(fanSpeed));
	    }
    }

    timer.Stop();
}

} // namespace ktt

#endif // KTT_POWER_USAGE_NVML
