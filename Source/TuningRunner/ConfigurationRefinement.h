#pragma once

#include <vector>
#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Configuration/RefinementParameters.h>
#include <Api/Output/KernelResult.h>
#include <KttTypes.h>

namespace ktt
{

class ConfigurationRefinement
{
public:
    explicit ConfigurationRefinement(const RefinementParameters& parameters);

    bool Start(const std::vector<std::pair<KernelConfiguration, Nanoseconds>>& history);

    KernelConfiguration GetKernelConfiguration() const;

    void AddResult(const KernelResult& result);

    bool IsFinished() const;

    KernelConfiguration GetWinner() const;

    std::vector<std::pair<KernelConfiguration, Nanoseconds>> GetRefinedResults() const;

    void Cancel();

private:
    struct Candidate
    {
        KernelConfiguration configuration;
        Nanoseconds explorationDuration;
        std::vector<Nanoseconds> refinementDurations;
        bool active;
    };

    RefinementParameters m_Parameters;
    std::vector<Candidate> m_Candidates;

    size_t m_CurrentRound;
    size_t m_CurrentCandidateIndex;
    bool m_MovingForward;
    bool m_Finished;
    KernelConfiguration m_Winner;

    void AdvancePosition();
    void ComputeWinner();
};

} // namespace ktt