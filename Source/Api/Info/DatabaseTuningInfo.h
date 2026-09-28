#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <ComputeEngine/ComputeApi.h>
#include <KttPlatform.h>

namespace ktt::db
{

/** @struct DeviceInfo
 * Contains information about a compute device (GPU or CPU).
 * Used to identify and characterize the hardware on which tuning was performed.
 */
struct DeviceInfo
{
    std::string name; ///< Device name (e.g., "GeForce RTX 3090")
    std::optional<std::string> deviceIdentifier; ///< Persistent hardware identifier of the device (e.g. device UUID), empty when the compute API does not expose one
    std::string vendor; ///< Device vendor (e.g., "NVIDIA", "AMD")
    std::string type; ///< Device type (e.g., "GPU", "CPU")
    std::string driverVersion; ///< Version of the device driver (e.g., "550.54.14"), empty when unknown or not applicable
    std::optional<std::string> extensions; ///< Supported device extensions (optional)
    std::optional<uint32_t>
        cudaComputeCapabilityMajor{}; ///< CUDA compute capability major version (optional, NVIDIA only)
    std::optional<uint32_t>
        cudaComputeCapabilityMinor{}; ///< CUDA compute capability minor version (optional, NVIDIA only)
    ComputeApi computeApi{ComputeApi::Cpp}; ///< Compute API used (CUDA, OpenCL, C++, etc.)
};

/** @struct TunerInfo
 * Identifies the tuner (and its version) that produced the tuning results.
 */
struct TunerInfo
{
    std::string name{"KTT"}; ///< Name of the tuner
    std::string version{GetKttVersionString()}; ///< Version of the tuner (e.g., "2.3.1")
};

/** @struct TuningSpaceInfo
 * Contains fingerprints (checksums) that uniquely identify tuning parameters, source code, and the tuning space.
 * Used for validation and to detect if tuning configurations have changed between executions.
 */
struct TuningSpaceInfo
{
    size_t parameterFingerprint{}; ///< Hash/fingerprint of tuning parameters
    size_t sourceFingerprint{}; ///< Hash/fingerprint of kernel source code
    size_t spaceFingerprint{}; ///< Hash/fingerprint of the entire tuning space configuration
};

/** @struct TuningInfo
 * Data transfer object that aggregates all information about tuning source and tuning space for which results were
 * saved to the database. Combines device characteristics, input data, and tuning space fingerprints.
 */
struct TuningInfo
{
    DeviceInfo device{}; ///< Information about the compute device used for tuning
    TunerInfo tuner{}; ///< Tuner that produced the results
    std::optional<std::string> inputData{}; ///< Optional input data configuration (e.g., dataset identifier)
    TuningSpaceInfo spaceInfo{}; ///< Fingerprints identifying the tuning space
};

} // namespace ktt::db
