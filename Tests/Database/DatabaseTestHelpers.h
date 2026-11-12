#pragma once

// Shared fixture helpers for the Database unit tests. This header is only included from within the
// #if defined(KTT_DATABASE) guard of each Database test file (the database extension is compiled in
// with premake --database, which defines KTT_DATABASE and pulls in the Database sources + sqlite3).
// The helpers are inline in a named namespace so the header can be included in several translation
// units without ODR issues and without unused-function warnings in files that use only some of them.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <Api/Configuration/ParameterPair.h>
#include <Api/Info/DatabaseTuningInfo.h>
#include <Api/KttException.h>
#include <Api/Output/ComputationResult.h>
#include <Api/Output/KernelResult.h>
#include <ComputeEngine/ComputeApi.h>
#include <Database/Database.h>

namespace ktt::db::test
{

// One whole millisecond in nanoseconds. The default KTT time unit is milliseconds, so kernel results
// serialize to and from JSON as whole-millisecond doubles; using whole milliseconds keeps durations
// exact across the store/load round-trip.
inline constexpr ktt::Nanoseconds Millisecond = 1'000'000;

// Builds a minimal but valid TuningInfo. The C++ (CPU) compute API is used because it requires neither
// device extensions nor CUDA compute capabilities, keeping the fixture small. See ValidateTuningInfo:
// all three fingerprints must be non-zero and the device name/type must be set.
inline ktt::db::TuningInfo MakeTuningInfo()
{
    ktt::db::TuningInfo info;
    info.device.name = "Test CPU";
    info.device.vendor = "Test Vendor";
    info.device.type = "CPU";
    info.device.computeApi = ktt::ComputeApi::Cpp;

    info.spaceInfo.parameterFingerprint = 111;
    info.spaceInfo.sourceFingerprint = 222;
    info.spaceInfo.spaceFingerprint = 333;
    return info;
}

// Builds a successful KernelResult carrying a single computation result of the given kernel duration.
// Only results with status Ok are persisted, and the stored duration is the sum of the computation
// result durations (here just one), which is what SimpleGetBestResults orders by.
inline ktt::KernelResult MakeResult(const std::string& kernelName, const uint64_t blockSize, const ktt::Nanoseconds duration)
{
    const ktt::KernelConfiguration configuration({ktt::ParameterPair("block_size", static_cast<uint64_t>(blockSize))});

    ktt::ComputationResult computation("kernelFunction");
    computation.SetDurationData(duration, 0, 0);
    computation.SetSizeData(ktt::DimensionVector(1024), ktt::DimensionVector(64));

    return ktt::KernelResult(kernelName, configuration, {computation}, "2026-09-09T00:00:00");
}

} // namespace ktt::db::test
