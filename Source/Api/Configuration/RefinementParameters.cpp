#include <Api/Configuration/RefinementParameters.h>

namespace ktt
{

RefinementParameters::RefinementParameters() :
    fraction(0.0),
    minCandidates(0),
    maxCandidates(0),
    minExplored(0),
    rounds(0),
    maxRelativeGap(std::nullopt)
{}

RefinementParameters::RefinementParameters(const double fraction, const uint64_t minCandidates,
    const uint64_t maxCandidates, const uint64_t minExplored, const uint64_t rounds,
    const std::optional<double> maxRelativeGap) :
    fraction(fraction),
    minCandidates(minCandidates),
    maxCandidates(maxCandidates),
    minExplored(minExplored),
    rounds(rounds),
    maxRelativeGap(maxRelativeGap)
{}

bool RefinementParameters::IsEnabled() const
{
    return fraction > 0.0 && maxCandidates > 0 && rounds > 0;
}

bool RefinementParameters::IsValid() const
{
    if (!IsEnabled())
    {
        return true;
    }

    return fraction > 0.0 && fraction < 1.0 &&
           minCandidates <= maxCandidates &&
           minCandidates > 0 && rounds > 0 &&
               (!maxRelativeGap.has_value() || maxRelativeGap.value() >= 0.0);
}
}
