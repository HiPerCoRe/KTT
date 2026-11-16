#pragma once

#include <cstdint>
#include <optional>
#include <KttPlatform.h>

namespace ktt
{

/** @struct RefinementParameters
  * Struct which holds parameters for dynamic tuning refinement phase.
  */
struct KTT_API RefinementParameters
{
    /** Fraction of total explored configurations to be selected for refinement (0.0 - 1.0). */
    double fraction;

    /** Minimum absolute number of candidates to select. */
    uint64_t minCandidates;

    /** Maximum absolute number of candidates to select. */
    uint64_t maxCandidates;

    /** Minimum number of valid explored configurations required to trigger refinement. */
    uint64_t minExplored;

    /** Number of measurement rounds for each candidate. */
    uint64_t rounds;

    /** Optional maximum relative time gap against the best configuration.
      * E.g., 0.1 means candidates can be at most 10% slower than the best configuration. */
    std::optional<double> maxRelativeGap;

    RefinementParameters();

    RefinementParameters(const double fraction, const uint64_t minCandidates, const uint64_t maxCandidates,
        const uint64_t minExplored, const uint64_t rounds, const std::optional<double> maxRelativeGap = std::nullopt);

    bool IsValid() const;
};

} // namespace ktt
