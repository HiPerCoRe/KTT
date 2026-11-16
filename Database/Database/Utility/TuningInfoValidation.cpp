#include <cstdint>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Api/KttException.h>
#include <Database/Utility/TuningInfoValidation.h>

namespace ktt::db
{

void ValidateTuningInfo(const TuningInfo &tuningInfo)
{
    const auto &spaceInfo = tuningInfo.spaceInfo;
    if (spaceInfo.parameterFingerprint == 0 || spaceInfo.sourceFingerprint == 0 || spaceInfo.spaceFingerprint == 0)
        throw KttException("Invalid TuningInfo: spaceInfo fingerprints must all be non-zero");

    if (tuningInfo.tuner.name.empty() || tuningInfo.tuner.version.empty())
        throw KttException("Invalid TuningInfo: tuner name and version must not be empty");

    const auto &device = tuningInfo.device;
    if (device.name.empty())
        throw KttException("Invalid TuningInfo: device name must not be empty");

    if (device.type.empty())
        throw KttException("Invalid TuningInfo: device type must not be empty");

    const bool hasCudaCapability =
        device.cudaComputeCapabilityMajor.has_value() && device.cudaComputeCapabilityMinor.has_value();
    const bool hasAnyCudaCapability =
        device.cudaComputeCapabilityMajor.has_value() || device.cudaComputeCapabilityMinor.has_value();
    const bool hasExtensions = device.extensions.has_value();

    switch (device.computeApi)
    {
    case ComputeApi::CUDA:
        if (!hasCudaCapability)
            throw KttException(
                "Invalid TuningInfo: CUDA device must set both CUDA compute capability major and minor versions"
            );
        if (hasExtensions)
            throw KttException("Invalid TuningInfo: CUDA device must not set extensions");
        break;

    case ComputeApi::OpenCL:
    case ComputeApi::Vulkan:
        if (!hasExtensions)
            throw KttException("Invalid TuningInfo: OpenCL/Vulkan device must set extensions");
        if (hasAnyCudaCapability)
            throw KttException("Invalid TuningInfo: OpenCL/Vulkan device must not set CUDA compute capabilities");
        break;

    case ComputeApi::Cpp:
        if (hasExtensions || hasAnyCudaCapability)
            throw KttException("Invalid TuningInfo: C++ device must not set extensions or CUDA compute capabilities");
        break;
    }
}

} // namespace ktt::db
