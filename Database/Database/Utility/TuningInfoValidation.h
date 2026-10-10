#pragma once

namespace ktt::db
{

struct TuningInfo;

/** @fn void ValidateTuningInfo(const TuningInfo& tuningInfo)
 * Validates a TuningInfo before it is used to store or query tuning results.
 * The following invariants are enforced:
 *  - No fingerprint in spaceInfo may be zero.
 *  - The device name and type must not be empty.
 *  - When the compute API is CUDA, both CUDA compute capability versions must be set and no extensions may be set.
 *  - When the compute API is OpenCL or Vulkan, only the extensions must be set (no CUDA compute capabilities).
 *  - When the compute API is C++, none of the extensions or CUDA compute capabilities may be set.
 * @param tuningInfo The TuningInfo to validate.
 * @throw KttException If any of the invariants above is violated.
 */
void ValidateTuningInfo(const TuningInfo &tuningInfo);

} // namespace ktt::db
