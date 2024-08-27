#include <algorithm>

#include <Api/KttException.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>
#include <TuningRunner/TuningRunner.h>
#include <Utility/Logger/Logger.h>
#include <Utility/Timer/ScopeTimer.h>

namespace ktt
{

TuningRunner::TuningRunner(KernelRunner& kernelRunner) :
    m_KernelRunner(kernelRunner),
    m_ConfigurationManager(std::make_unique<ConfigurationManager>())
{}

std::vector<KernelResult> TuningRunner::Tune(const Kernel& kernel, const KernelDimensions& dimensions,
    std::unique_ptr<StopCondition> stopCondition)
{
    Logger::LogInfo("Starting offline tuning for kernel " + kernel.GetName());
    const auto id = kernel.GetId();

    if (!m_ConfigurationManager->HasData(id))
    {
        m_ConfigurationManager->InitializeData(kernel);
    }

    if (stopCondition != nullptr)
    {
        const uint64_t configurationsCount = m_ConfigurationManager->GetTotalConfigurationsCount(id);
        stopCondition->Initialize(configurationsCount);
    }

    std::vector<KernelResult> results;
//    KernelResult result(kernel.GetName(), m_ConfigurationManager->GetCurrentConfiguration(id));

    while (!m_ConfigurationManager->IsDataProcessed(id))
    {
        KernelResult result(kernel.GetName(), m_ConfigurationManager->GetCurrentConfiguration(id));
        KernelResult multiResult(kernel.GetName(), m_ConfigurationManager->GetCurrentConfiguration(id));
        int iter = 0;
        do 
        {
            result = TuneIteration(kernel, dimensions, KernelRunMode::OfflineTuning, std::vector<BufferOutputDescriptor>{}, false);
            multiResult.FuseProfilingTimes(result, (iter == 0));
            iter++;
        }
        while (result.HasRemainingProfilingRuns());
        if (iter > 1) //do not copy the same result twice
            result.CopyProfilingTimes(multiResult);

        const Nanoseconds searcherOverhead = RunScopeTimer([this, id, &result]()
        {
            m_ConfigurationManager->CalculateNextConfiguration(id, result);
        });

        result.SetSearcherOverhead(searcherOverhead);
        results.push_back(result);

        if (stopCondition == nullptr)
        {
            continue;
        }

        stopCondition->Update(result);
        Logger::LogInfo(stopCondition->GetStatusString());

        if (stopCondition->IsFulfilled())
        {
            break;
        }
    }

    Logger::LogInfo("Ending offline tuning for kernel " + kernel.GetName() + ", total number of tested configurations is "
        + std::to_string(results.size()));
    m_KernelRunner.ClearReferenceResult(kernel);
    return results;
}

KernelResult TuningRunner::TuneIteration(const Kernel& kernel, const KernelDimensions& dimensions, const KernelRunMode mode,
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

    KernelResult result = m_KernelRunner.RunKernel(kernel, configuration, dimensions, mode, output);

    if (mode != KernelRunMode::OfflineTuning && !result.HasRemainingProfilingRuns() && !m_ConfigurationManager->IsDataProcessed(id))
    {
        const Nanoseconds searcherOverhead = RunScopeTimer([this, id, &result]()
        {
            m_ConfigurationManager->CalculateNextConfiguration(id, result);
        });

        result.SetSearcherOverhead(searcherOverhead);
    }

    return result;
}

std::vector<KernelResult> TuningRunner::SimulateTuning(const Kernel& kernel, const std::vector<KernelResult>& results,
    std::unique_ptr<StopCondition> stopCondition)
{
    Logger::LogInfo("Starting simulated tuning for kernel " + kernel.GetName());
    const auto id = kernel.GetId();

    if (!m_ConfigurationManager->HasData(id))
    {
        m_ConfigurationManager->InitializeData(kernel);
    }

    uint64_t passedIterations = 0;
    std::vector<KernelResult> output;
    const uint64_t configurationCount = m_ConfigurationManager->GetTotalConfigurationsCount(id);

    if (stopCondition != nullptr)
    {
        stopCondition->Initialize(configurationCount);
    }

    while (!m_ConfigurationManager->IsDataProcessed(id))
    {
        if (stopCondition != nullptr && stopCondition->IsFulfilled())
        {
            break;
        }

        const auto currentConfiguration = m_ConfigurationManager->GetCurrentConfiguration(id);
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

        const Nanoseconds searcherOverhead = RunScopeTimer([this, id, &result]()
        {
            m_ConfigurationManager->CalculateNextConfiguration(id, result);
        });

        result.SetSearcherOverhead(searcherOverhead);
        output.push_back(result);
        ++passedIterations;

        if (stopCondition != nullptr)
        {
            stopCondition->Update(result);
            Logger::LogInfo(stopCondition->GetStatusString());
        }
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

void TuningRunner::InitializeConfigurationData(const Kernel& kernel)
{
    m_ConfigurationManager->InitializeData(kernel);
}

void TuningRunner::ClearConfigurationData(const KernelId id, const bool clearSearcher)
{
    m_ConfigurationManager->ClearData(id, clearSearcher);
}

uint64_t TuningRunner::GetConfigurationsCount(const KernelId id) const
{
    return m_ConfigurationManager->GetTotalConfigurationsCount(id);
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
