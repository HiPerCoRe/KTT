#include <TuningRunner/ConfigurationRefinement.h>
#include <Api/Configuration/PreciseMeasurementParameters.h>
#include <Utility/Logger/Logger.h>
#include <algorithm>
#include <cmath>

namespace ktt
{
    ConfigurationRefinement::ConfigurationRefinement(const RefinementParameters& parameters) :
        m_Parameters(parameters),
        m_CurrentRound(0),
        m_CurrentCandidateIndex(0),
        m_MovingForward(true),
        m_Finished(false)
    {}

    bool ConfigurationRefinement::Start(const std::vector<std::pair<KernelConfiguration, Nanoseconds>>& history)
    {

        if (!m_Parameters.IsValid())
        {
            Logger::LogWarning("Refinement parameters are invalid, skipping refinement");
            m_Finished = true;
            return false;
        }

        std::vector<std::pair<KernelConfiguration, std::vector<Nanoseconds>>> grouped;

        for (const auto& [configuration, duration] : history)
        {
            auto it = std::find_if(grouped.begin(), grouped.end(), [&configuration](const auto& entry)
            {
                return entry.first == configuration;
            });

            if (it == grouped.end())
            {
                grouped.push_back({configuration, {duration}});
            }
            else
            {
                it->second.push_back(duration);
            }
        }

        std::vector<std::pair<KernelConfiguration, Nanoseconds>> uniqueHistory;
        uniqueHistory.reserve(grouped.size());

        for (const auto& [configuration, durations] : grouped)
        {
            uniqueHistory.push_back({configuration, PreciseMeasurementParameters::CalculateDuration(durations, DurationCalculationMethod::Median)});
        }

        if (uniqueHistory.size() < history.size())
        {
            Logger::LogInfo("Refinement merged " + std::to_string(history.size() - uniqueHistory.size()) + " duplicate configurations from exploration history");
        }

        if (uniqueHistory.size() < m_Parameters.minExplored)
        {
            Logger::LogInfo("Refinement skipped, not enough configurations");
            m_Finished = true;
            return false;
        }

        auto sortedHistory = uniqueHistory;
        std::sort(sortedHistory.begin(), sortedHistory.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        size_t candidateCount = static_cast<size_t>(std::ceil(m_Parameters.fraction * static_cast<double>(uniqueHistory.size())));
        candidateCount = std::max(candidateCount, static_cast<size_t>(m_Parameters.minCandidates));
        candidateCount = std::min(candidateCount, static_cast<size_t>(m_Parameters.maxCandidates));
        candidateCount = std::min(candidateCount, uniqueHistory.size());

        if (m_Parameters.maxRelativeGap.has_value() && !sortedHistory.empty())
        {
            const Nanoseconds bestDuration = sortedHistory.front().second;
            const Nanoseconds limit = static_cast<Nanoseconds>(bestDuration * (1.0 + m_Parameters.maxRelativeGap.value()));

            size_t filteredCount = 0;
            for (size_t i = 0; i < candidateCount; ++i)
            {
                if (sortedHistory[i].second <= limit)
                {
                    filteredCount++;
                }
                else
                {
                    break;
                }
            }
            candidateCount = filteredCount;
        }

        if (candidateCount < 2)
        {
            Logger::LogInfo("Not enough candidates, skipping refinement");
            m_Finished = true;
            return false;
        }

        m_Candidates.reserve(candidateCount);
        for (size_t i = 0; i < candidateCount; ++i)
        {
            m_Candidates.push_back({sortedHistory[i].first, sortedHistory[i].second, {}, true});
        }

        Logger::LogInfo("Beginning refinement, Candidates: " + std::to_string(candidateCount) + ", rounds: " + std::to_string(m_Parameters.rounds) + ".");
        Logger::LogInfo("Refinement - Round: 1/" + std::to_string(m_Parameters.rounds) + ", Candidate: 1/" + std::to_string(m_Candidates.size()));

        return true;
    }

    KernelConfiguration ConfigurationRefinement::GetKernelConfiguration() const
    {
        return m_Candidates[m_CurrentCandidateIndex].configuration;
    }

    void ConfigurationRefinement::AddResult(const KernelResult& result)
    {
        if (m_Finished)
        {
            return;
        }

        if (result.GetStatus() == ResultStatus::Ok)
        {
            m_Candidates[m_CurrentCandidateIndex].refinementDurations.push_back(result.GetTotalDuration());
        }
        else
        {
            Logger::LogWarning("Candidate failed");
            m_Candidates[m_CurrentCandidateIndex].active = false;
        }

        AdvancePosition();
    }

    bool ConfigurationRefinement::IsFinished() const
    {
        return m_Finished;
    }

    KernelConfiguration ConfigurationRefinement::GetWinner() const
    {
        return m_Winner;
    }

    void ConfigurationRefinement::Cancel()
    {
        m_Finished = true;
    }

    void ConfigurationRefinement::AdvancePosition()
    {
        bool foundNext = false;

        while (!foundNext)
        {
            if (m_MovingForward)
            {
                if (m_CurrentCandidateIndex + 1 < m_Candidates.size())
                {
                    m_CurrentCandidateIndex++;
                }
                else
                {
                    m_CurrentRound++;
                    m_MovingForward = false;
                }
            }
            else
            {
                if (m_CurrentCandidateIndex > 0)
                {
                    m_CurrentCandidateIndex--;
                }
                else
                {
                    m_CurrentRound++;
                    m_MovingForward = true;
                }
            }

            if (m_CurrentRound >= m_Parameters.rounds)
            {
                ComputeWinner();
                m_Finished = true;
                return;
            }

            if (m_Candidates[m_CurrentCandidateIndex].active)
            {
                foundNext = true;
            }
            else
            {
                bool anyActive = false;
                for (const auto& candidate : m_Candidates)
                {
                    if (candidate.active)
                    {
                        anyActive = true;
                        break;
                    }
                }

                if (!anyActive)
                {
                    Logger::LogWarning("All candidates failed refinement");
                    ComputeWinner();
                    m_Finished = true;
                    return;
                }
            }
        }

        Logger::LogInfo("Refinement - Round: " + std::to_string(m_CurrentRound + 1) + "/" + std::to_string(m_Parameters.rounds) +
                    ", Candidate: " + std::to_string(m_CurrentCandidateIndex + 1) + "/" + std::to_string(m_Candidates.size()));
    }

    void ConfigurationRefinement::ComputeWinner()
    {
        Nanoseconds bestMedian = InvalidDuration;
        const Candidate* bestCandidate = nullptr;

        for (const auto& candidate : m_Candidates)
        {
            if (!candidate.active || candidate.refinementDurations.empty())
            {
                continue;
            }

            const Nanoseconds median = PreciseMeasurementParameters::CalculateDuration(candidate.refinementDurations, DurationCalculationMethod::Median);

            if (bestMedian == InvalidDuration || median < bestMedian)
            {
                bestMedian = median;
                bestCandidate = &candidate;
            }
            else if (median == bestMedian)
            {
                if (candidate.explorationDuration < bestCandidate->explorationDuration)
                {
                    bestCandidate = &candidate;
                }
            }
        }

        if (bestCandidate != nullptr)
        {
            m_Winner = bestCandidate->configuration;
        }
        else
        {
            m_Winner = m_Candidates.front().configuration;
        }
    }
} // namespace ktt
