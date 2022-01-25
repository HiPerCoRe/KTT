#include <chrono>
#include <limits>
#include <string>

#include <Api/Searcher/McmcSearcher.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

McmcSearcher::McmcSearcher(const KernelConfiguration& start) :
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

McmcSearcher::McmcSearcher([[maybe_unused]] const std::vector<double>& start) :
    McmcSearcher(KernelConfiguration())
{}

void McmcSearcher::OnInitialize()
{
    m_IntDistribution = std::uniform_int_distribution<size_t>(0, GetConfigurationsCount() - 1);
    size_t initialState = 0;

    if (m_Start.IsValid())
    {
        initialState = GetIndex(m_Start);
    } 
    else
    {
        initialState = m_IntDistribution(m_Generator);
        m_Boot = m_BootIterations;
    }

    m_OriginState = initialState;
    m_CurrentState = initialState;
    m_Index = initialState;
}

void McmcSearcher::OnReset()
{
    m_Index = 0;
    m_VisitedStatesCount = 0;
    m_OriginState = 0;
    m_CurrentState = 0;
    m_BestTime = std::numeric_limits<double>::max();
    m_ExecutionTimes.clear();
}

bool McmcSearcher::CalculateNextConfiguration(const KernelResult& previousResult)
{
    ++m_VisitedStatesCount;

    if (previousResult.IsValid())
    {
        m_ExecutionTimes[m_Index] = static_cast<double>(previousResult.GetTotalDuration());
    }
    else
    {
        m_ExecutionTimes[m_Index] = std::numeric_limits<double>::max();
    }

    // boot-up, sweeps randomly across bootIterations states and sets
    // origin of MCMC to the best state
    if (m_Boot > 0) 
    {
        if (m_ExecutionTimes.find(m_CurrentState) == m_ExecutionTimes.cend())
        {
            m_ExecutionTimes[m_CurrentState] = std::numeric_limits<double>::max();
        }

        if (m_ExecutionTimes.find(m_OriginState) == m_ExecutionTimes.cend())
        {
            m_ExecutionTimes[m_OriginState] = std::numeric_limits<double>::max();
        }

        if (m_ExecutionTimes[m_CurrentState] <= m_ExecutionTimes[m_OriginState])
        {            
            m_OriginState = m_CurrentState;
            m_BestTime = m_ExecutionTimes[m_CurrentState];
            Logger::LogDebug("MCMC boot step " + std::to_string(m_VisitedStatesCount) + ", new best performance: "
                + std::to_string(m_BestTime));
        }

        --m_Boot;

        if (GetExploredIndices().size() == GetConfigurationsCount())
        {
            return false;
        }

        while (GetExploredIndices().find(m_Index) != GetExploredIndices().cend())
        {
            m_Index = m_IntDistribution(m_Generator);
        }

        m_CurrentState = m_Index;
        return true;
    }

    // acceptation of a new state
    if (m_ExecutionTimes.find(m_CurrentState) == m_ExecutionTimes.cend())
    {
        m_ExecutionTimes[m_CurrentState] = std::numeric_limits<double>::max();
    }

    if (m_ExecutionTimes.find(m_OriginState) == m_ExecutionTimes.cend())
    {
        m_ExecutionTimes[m_OriginState] = std::numeric_limits<double>::max();
    }

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

    if (GetExploredIndices().size() == GetConfigurationsCount())
    {
        return false;
    }

    std::vector<KernelConfiguration> neighbourConfigurations = GetNeighbourConfigurations(GetConfiguration(m_OriginState),
        m_MaximumDifferences, std::numeric_limits<size_t>::max());
    std::vector<size_t> neighbours;

    for (const auto& neighbour : neighbourConfigurations)
    {
        const size_t index = GetIndex(neighbour);
        neighbours.push_back(index);
    }

    // reset origin position when there are no neighbours
    if (neighbours.empty())
    {
        Logger::LogDebug("MCMC step " + std::to_string(m_VisitedStatesCount) + ", no neighbours, resetting position");
        
        while (GetExploredIndices().find(m_OriginState) != GetExploredIndices().cend())
        {
            m_OriginState = m_IntDistribution(m_Generator);
        }
        
        m_Index = m_OriginState;
        m_CurrentState = m_OriginState;
        return true;
    }

    Logger::LogDebug("MCMC step " + std::to_string(m_VisitedStatesCount) + ", choosing randomly one of "
        + std::to_string(neighbours.size()) + " neighbours");

    // select a random neighbour state
    m_CurrentState = neighbours.at(m_IntDistribution(m_Generator) % neighbours.size());
    m_Index = m_CurrentState;
    return true;
}

KernelConfiguration McmcSearcher::GetCurrentConfiguration() const
{
    return GetConfiguration(m_Index);
}

} // namespace ktt
