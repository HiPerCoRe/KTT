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
}

NvmlPowerManager::~NvmlPowerManager()
{
    CheckError(nvmlShutdown(), "nvmlShutdown");
}

void NvmlPowerManager::StartCollection()
{
    m_Context.Synchronize();

    m_PowerSamples.clear();
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

void NvmlPowerManager::CollectPowerSamples()
{
    uint32_t initialValue;
    CheckError(nvmlDeviceGetPowerUsage(m_Device, &initialValue), "nvmlDeviceGetPowerUsage");
    m_PowerSamples.push_back(initialValue);

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
    }

    timer.Stop();
}

} // namespace ktt

#endif // KTT_POWER_USAGE_NVML
