#include <algorithm>

#include <Api/KttException.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <TuningRunner/TuningRunner.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

TuningRunner::TuningRunner(KernelRunner& kernelRunner) :
    m_KernelRunner(kernelRunner),
    m_ConfigurationManager(std::make_unique<ConfigurationManager>())
{}

std::vector<KernelResult> TuningRunner::Tune(const Kernel& kernel, std::unique_ptr<StopCondition> stopCondition)
{
    Logger::LogInfo("Starting offline tuning for kernel " + kernel.GetName());
    m_ConfigurationManager->InitializeData(kernel);
    const auto id = kernel.GetId();

    if (stopCondition != nullptr)
    {
        const uint64_t configurationsCount = m_ConfigurationManager->GetTotalConfigurationsCount(id);
        stopCondition->Initialize(configurationsCount);
    }

    std::vector<KernelResult> results;
    KernelResult result(kernel.GetName(), m_ConfigurationManager->GetCurrentConfiguration(id));

    while (!m_ConfigurationManager->IsDataProcessed(id))
    {
        do 
        {
            result = TuneIteration(kernel, KernelRunMode::OfflineTuning, std::vector<BufferOutputDescriptor>{}, false);
        }
        while (result.HasRemainingProfilingRuns());

        results.push_back(result);

        if (stopCondition != nullptr)
        {
            stopCondition->Update(result);
            Logger::LogInfo(stopCondition->GetStatusString());

            if (stopCondition->IsFulfilled())
            {
                break;
            }
        }

        m_ConfigurationManager->CalculateNextConfiguration(id, result);
    }

    Logger::LogInfo("Ending offline tuning for kernel " + kernel.GetName() + ", total number of tested configurations is "
        + std::to_string(results.size()));
    m_KernelRunner.ClearReferenceResult(kernel);
    m_ConfigurationManager->ClearData(id);
    return results;
}

KernelResult TuningRunner::TuneIteration(const Kernel& kernel, const KernelRunMode mode,
    const std::vector<BufferOutputDescriptor>& output, const bool recomputeReference)
{
    if (recomputeReference)
    {
        m_KernelRunner.ClearReferenceResult(kernel);
    }

    const auto id = kernel.GetId();

    if (!m_ConfigurationManager->HasData(id))
    {
        m_ConfigurationManager->InitializeData(kernel);
    }

    KernelConfiguration configuration;

    if (m_ConfigurationManager->IsDataProcessed(id))
    {
        configuration = m_ConfigurationManager->GetBestConfiguration(id);
        Logger::LogInfo("Launching the best configuration for kernel " + kernel.GetName());
    }
    else
    {
        configuration = m_ConfigurationManager->GetCurrentConfiguration(id);

        const uint64_t configurationNumber = m_ConfigurationManager->GetExploredConfigurationsCount(id) + 1;
        const uint64_t configurationCount = m_ConfigurationManager->GetTotalConfigurationsCount(id);
        Logger::LogInfo("Launching configuration " + std::to_string(configurationNumber) + " / " + std::to_string(configurationCount)
            + " for kernel " + kernel.GetName());
    }

    KernelResult result = m_KernelRunner.RunKernel(kernel, configuration, mode, output);

    if (mode != KernelRunMode::OfflineTuning && !result.HasRemainingProfilingRuns() && !m_ConfigurationManager->IsDataProcessed(id))
    {
        m_ConfigurationManager->CalculateNextConfiguration(id, result);
    }

    return result;
}

std::vector<KernelResult> TuningRunner::SimulateTuning(const Kernel& kernel, const std::vector<KernelResult>& results,
    const uint64_t iterations)
{
    Logger::LogInfo("Starting simulated tuning for kernel " + kernel.GetName());
    const auto id = kernel.GetId();

    if (!m_ConfigurationManager->HasData(id))
    {
        m_ConfigurationManager->InitializeData(kernel);
    }

    uint64_t passedIterations = 0;
    std::vector<KernelResult> output;

    while (!m_ConfigurationManager->IsDataProcessed(id))
    {
        if (iterations != 0 && passedIterations >= iterations)
        {
            break;
        }

        const auto currentConfiguration = m_ConfigurationManager->GetCurrentConfiguration(id);
        const uint64_t configurationCount = iterations != 0 ? iterations : m_ConfigurationManager->GetTotalConfigurationsCount(id);
        KernelResult result;

        try
        {
            Logger::LogInfo("Simulating run for configuration " + std::to_string(passedIterations + 1) + " / "
                + std::to_string(configurationCount) + " for kernel " + kernel.GetName() + ": " + currentConfiguration.GetString());
            result = FindMatchingResult(results, currentConfiguration);

            const auto& time = TimeConfiguration::GetInstance();
            const uint64_t duration = time.ConvertFromNanoseconds(result.GetTotalDuration());
            const uint64_t kernelDuration = time.ConvertFromNanoseconds(result.GetKernelDuration());
            const std::string tag = time.GetUnitTag();
            Logger::LogInfo("Kernel run completed successfully in " + std::to_string(duration) + tag + ", kernel duration was "
                + std::to_string(kernelDuration) + tag);
        }
        catch (const KttException& error)
        {
            Logger::LogWarning(std::string("Kernel run failed, reason: ") + error.what());
            result = KernelResult(kernel.GetName(), currentConfiguration);
            result.SetStatus(ResultStatus::ComputationFailed);
        }

        ++passedIterations;
        m_ConfigurationManager->CalculateNextConfiguration(id, result);
        output.push_back(result);
    }

    Logger::LogInfo("Ending simulated tuning for kernel " + kernel.GetName() + ", total number of tested configurations is "
        + std::to_string(passedIterations));
    m_ConfigurationManager->ClearData(id);
    return output;
}

void TuningRunner::SetSearcher(const KernelId id, std::unique_ptr<Searcher> searcher)
{
    m_ConfigurationManager->SetSearcher(id, std::move(searcher));
}

void TuningRunner::ClearData(const KernelId id, const bool clearSearcher)
{
    m_ConfigurationManager->ClearData(id, clearSearcher);
}

KernelConfiguration TuningRunner::GetBestConfiguration(const KernelId id) const
{
    return m_ConfigurationManager->GetBestConfiguration(id);
}

const KernelResult& TuningRunner::FindMatchingResult(const std::vector<KernelResult>& results,
    const KernelConfiguration& configuration)
{
    for (const auto& result : results)
    {
        if (result.GetConfiguration() == configuration)
        {
            return result;
        }
    }

    throw KttException("Matching configuration was not found");
}

} // namespace ktt
