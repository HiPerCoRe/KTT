#include <chrono>
#include <limits>
#include <string>

#include <Api/Searcher/McmcSearcher.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

McmcSearcher::McmcSearcher(const std::vector<double>& start) :
    Searcher(),
    m_Index(0),
    m_VisitedStatesCount(0),
    m_OriginState(0),
    m_CurrentState(0),
    m_Boot(0),
    m_BestTime(std::numeric_limits<double>::max()),
    m_Start(start),
    m_Generator(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())),
    m_ProbabilityDistribution(0.0, 1.0)
{}

void McmcSearcher::OnInitialize()
{
    m_IntDistribution = std::uniform_int_distribution<size_t>(0, GetConfigurationsCount() - 1),
    m_ExecutionTimes.resize(GetConfigurationsCount(), std::numeric_limits<double>::max());

    size_t initialState = 0;

    if (!m_Start.empty())
    {
        initialState = SearchStateIndex(m_Start);
    } 
    else
    {
        initialState = m_IntDistribution(m_Generator);
        m_Boot = m_BootIterations;
    }

    m_OriginState = initialState;
    m_CurrentState = initialState;
    m_Index = initialState;

    for (size_t i = 0; i < GetConfigurationsCount(); ++i)
    {
        m_UnexploredIndices.insert(i);
    } 
}

void McmcSearcher::OnReset()
{
    m_Index = 0;
    m_VisitedStatesCount = 0;
    m_OriginState = 0;
    m_CurrentState = 0;
    m_BestTime = std::numeric_limits<double>::max();
    m_ExecutionTimes.clear();
    m_UnexploredIndices.clear();
}

void McmcSearcher::CalculateNextConfiguration(const KernelResult& previousResult)
{
    ++m_VisitedStatesCount;
    m_UnexploredIndices.erase(m_Index);
    m_ExecutionTimes[m_Index] = static_cast<double>(previousResult.GetTotalDuration());

    // boot-up, sweeps randomly across bootIterations states and sets
    // origin of MCMC to the best state
    if (m_Boot > 0) 
    {
        if (m_ExecutionTimes[m_CurrentState] <= m_ExecutionTimes[m_OriginState])
        {            
            m_OriginState = m_CurrentState;
            m_BestTime = m_ExecutionTimes[m_CurrentState];
            Logger::LogDebug("MCMC boot step " + std::to_string(m_VisitedStatesCount) + ", new best performance: "
                + std::to_string(m_BestTime));
        }

        --m_Boot;

        while (m_UnexploredIndices.find(m_Index) == m_UnexploredIndices.cend() || m_UnexploredIndices.empty()) 
        {
            m_Index = m_IntDistribution(m_Generator);
        }

        m_CurrentState = m_Index;
        return;
    }

    // acceptation of a new state
    if ((m_ExecutionTimes[m_CurrentState] <= m_ExecutionTimes[m_OriginState])
        || m_ProbabilityDistribution(m_Generator) < m_EscapeProbability)
    {
        m_OriginState = m_CurrentState;
            
        if (m_ExecutionTimes[m_CurrentState] < m_BestTime)
        {
            m_BestTime = m_ExecutionTimes[m_CurrentState];
        }

        if (m_ExecutionTimes[m_CurrentState] <= m_ExecutionTimes[m_OriginState])
        {
            Logger::LogDebug("MCMC step " + std::to_string(m_VisitedStatesCount)
                + ", accepting new state (performance improvement)");

            if (m_ExecutionTimes[m_CurrentState] == m_BestTime)
            {
                Logger::LogDebug("MCMC step " + std::to_string(m_VisitedStatesCount) + ", new best performance "
                    + std::to_string(m_BestTime));
            }
        }
        else
        {
            Logger::LogDebug("MCMC step " + std::to_string(m_VisitedStatesCount) + ", accepting new state (random escape)");
        }
    }
    else
    {
        Logger::LogDebug("MCMC step " + std::to_string(m_VisitedStatesCount) + ", continuing searching neighbours");
    }

    if (m_UnexploredIndices.empty())
    {
        return;
    }

    std::vector<size_t> neighbours = GetNeighbours(m_OriginState);

    // reset origin position when there are no neighbours
    if (neighbours.empty())
    {
        Logger::LogDebug("MCMC step " + std::to_string(m_VisitedStatesCount) + ", no neighbours, resetting position");

        while (m_UnexploredIndices.find(m_OriginState) == m_UnexploredIndices.cend())
        {
            m_OriginState = m_IntDistribution(m_Generator);
        }
        
        m_Index = m_OriginState;
        m_CurrentState = m_OriginState;
        return;
    }

    Logger::LogDebug("MCMC step " + std::to_string(m_VisitedStatesCount) + ", choosing randomly one of "
        + std::to_string(neighbours.size()) + " neighbours");

    // select a random neighbour state
    m_CurrentState = neighbours.at(m_IntDistribution(m_Generator) % neighbours.size());
    m_Index = m_CurrentState;
}

KernelConfiguration McmcSearcher::GetCurrentConfiguration() const
{
    return GetConfiguration(m_Index);
}

std::vector<size_t> McmcSearcher::GetNeighbours(const size_t referenceId) const
{
    std::vector<size_t> neighbours;
    const auto referenceConfiguration = GetConfiguration(referenceId);
    const auto& referencePairs = referenceConfiguration.GetPairs();

    for (const auto i : m_UnexploredIndices)
    {
        size_t differences = 0;
        size_t settingId = 0;
        const auto configuration = GetConfiguration(i);

        for (const auto& parameter : configuration.GetPairs())
        {
            if (!parameter.HasSameValue(referencePairs[settingId]))
            {
                ++differences;
            }

            ++settingId;
        }

        if (differences <= m_MaximumDifferences) 
        {
            neighbours.push_back(i);
        }
    }

    return neighbours;
}

size_t McmcSearcher::SearchStateIndex(const std::vector<double>& state) const
{
    size_t states = state.size();
    size_t ret = 0;
    bool match = true;

    for (uint64_t index = 0; index < GetConfigurationsCount(); ++index)
    {
        const auto configuration = GetConfiguration(index);
        match = true;

        for (size_t i = 0; i < states; ++i)
        {
            const auto& pair = configuration.GetPairs()[i];

            if ((pair.HasValueDouble() && pair.GetValueDouble() != state[i])
                || (pair.GetValue() != static_cast<uint64_t>(state[i])))
            {
                match = false;
                break;
            }
        }

        if (match)
        {
            break;
        }

        ++ret;
    }

    if (!match)
    {
        Logger::LogWarning("MCMC starting point not found.");
        ret = 0;
    }

    return ret;
}

} // namespace ktt
