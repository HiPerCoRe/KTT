#pragma once

#include <KttPlatform.h>

namespace ktt
{

/** @enum TuningPhase
 * Enum which specifies the current phase of the tuning process
 */
enum class KTT_API TuningPhase
{
    Exploration,
    Refinement,
    Finished
};

} // namespace ktt
