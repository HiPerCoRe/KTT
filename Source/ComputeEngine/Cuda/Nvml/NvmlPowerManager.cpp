#if defined(KTT_POWER_USAGE_NVML)

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
    m_SamplingInterval(static_cast<Nanoseconds>(1.0 / static_cast<double>(samplingFrequency) * 1'000'000'000.0))
{
    CheckError(nvmlInit_v2(), "nvmlInit_v2");
    CheckError(nvmlDeviceGetHandleByIndex(device, &m_Device), "nvmlDeviceGetHandleByIndex");
    m_PowerSamples.reserve(samplingFrequency);
    m_TempSamples.reserve(samplingFrequency);
    m_SMFreqSamples.reserve(samplingFrequency);
    m_MemFreqSamples.reserve(samplingFrequency);
    m_FanSpeedSamples.reserve(samplingFrequency);
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
    m_StopFlag = false;
    
    m_Future = m_Pool.push([this]()
    {
        CollectPowerSamples();
    });
}

void NvmlPowerManager::EndCollection()
{
    m_Context.Synchronize();

    m_StopFlag = true;
    m_Future.wait();
}

uint32_t NvmlPowerManager::GetPowerUsage() const
{
    Logger::LogDebug("Generating average power usage from number of samples: " + std::to_string(m_PowerSamples.size()));

    if (m_PowerSamples.empty())
    {
        return 0;
    }

    const uint32_t sum = std::accumulate(m_PowerSamples.cbegin(), m_PowerSamples.cend(), 0);
    return sum / static_cast<uint32_t>(m_PowerSamples.size());
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
    Logger::LogDebug("Generating average temperatures from number of samples: " + std::to_string(m_TempSamples.size()));

    if (m_TempSamples.empty())
    {
        return 0;
    }

    const double sum = std::accumulate(m_TempSamples.cbegin(), m_TempSamples.cend(), 0);
    return sum / static_cast<double>(m_TempSamples.size());
}

uint32_t NvmlPowerManager::GetSMFrequency() const
{
    Logger::LogDebug("Generating average SM frequency from number of samples: " + std::to_string(m_SMFreqSamples.size()));

    if (m_SMFreqSamples.empty())
    {
        return 0;
    }

    const uint32_t sum = std::accumulate(m_SMFreqSamples.cbegin(), m_SMFreqSamples.cend(), 0);
    return sum / static_cast<uint32_t>(m_SMFreqSamples.size());
}

uint32_t NvmlPowerManager::GetMemoryFrequency() const
{
    Logger::LogDebug("Generating average memory frequency from number of samples: " + std::to_string(m_SMFreqSamples.size()));

    if (m_MemFreqSamples.empty())
    {
        return 0;
    }

    const uint32_t sum = std::accumulate(m_MemFreqSamples.cbegin(), m_MemFreqSamples.cend(), 0);
    return sum / static_cast<uint32_t>(m_MemFreqSamples.size());
}

int32_t NvmlPowerManager::GetFanSpeed() const
{
    Logger::LogDebug("Generating average fan speed from number of samples: " + std::to_string(m_FanSpeedSamples.size()));

    if (m_FanSpeedSamples.empty())
    {
        return -1;
    }

    // If first sample is -1, fan speed is not supported by the device
    if (m_FanSpeedSamples[0] == -1)
    {
        return -1;
    }

    // Calculate average of all samples
    const int64_t sum = std::accumulate(m_FanSpeedSamples.cbegin(), m_FanSpeedSamples.cend(), static_cast<int64_t>(0));
    return static_cast<int32_t>(sum / static_cast<int64_t>(m_FanSpeedSamples.size()));
}

void NvmlPowerManager::CollectPowerSamples()
{
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
