#include <catch.hpp>

// Needs both databases: built with --database and --remote-database.
#if defined(KTT_REMOTE_DATABASE) && defined(KTT_DATABASE)

// R8.2: the same saves and reads on the local Database and on a RemoteDatabase backed by an in-memory fake ktt-db
// server must give the same answers.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <Api/KttException.h>
#include <Database/Local/Database.h>
#include <Database/Remote/RemoteDatabase.h>
#include <Database/Utility/TimeUnitGuard.h>

#include "../DatabaseTestHelpers.h"
#include "FakeKttDbServer.h"

using ktt::db::GetBestResultsQuery;
using ktt::db::IDatabase;
using ktt::db::test::FakeKttDbServer;
using ktt::db::test::MakeResult;
using ktt::db::test::MakeTuningInfo;
using ktt::db::test::Millisecond;

namespace
{

struct Pair
{
    std::shared_ptr<FakeKttDbServer> server = std::make_shared<FakeKttDbServer>(ktt::db::RemoteDatabase::GetHost());
    ktt::db::Database local{std::filesystem::path(":memory:")};
    ktt::db::RemoteDatabase remote{"kttak-parity", {}, server};

    std::vector<const IDatabase *> Both() const
    {
        return {&local, &remote};
    }

    void Save(const ktt::db::TuningInfo &info, const std::vector<ktt::KernelResult> &results,
        const ktt::OutputFormat format = ktt::OutputFormat::JSON) const
    {
        for (const IDatabase *db : Both())
            db->SaveResults(info, results, {format, 2});
    }
};

// What a caller can see of a result.
struct View
{
    std::string kernel;
    ktt::Nanoseconds duration;
    ktt::Nanoseconds totalDuration;
    std::string configuration;

    bool operator==(const View &other) const
    {
        return kernel == other.kernel && duration == other.duration && totalDuration == other.totalDuration
            && configuration == other.configuration;
    }
};

std::ostream &operator<<(std::ostream &stream, const View &view)
{
    return stream << view.kernel << " " << view.duration << "/" << view.totalDuration << " " << view.configuration;
}

std::vector<View> Views(const std::vector<ktt::KernelResult> &results)
{
    std::vector<View> views;

    for (const auto &result : results)
        views.push_back(View{result.GetKernelName(), result.GetKernelDuration(), result.GetTotalDuration(),
            result.GetConfiguration().GetString()});

    return views;
}

ktt::db::DeviceInfo CudaDevice(const std::string &name, const uint32_t major, const uint32_t minor)
{
    ktt::db::DeviceInfo device;
    device.name = name;
    device.vendor = "NVIDIA";
    device.type = "GPU";
    device.computeApi = ktt::ComputeApi::CUDA;
    device.cudaComputeCapabilityMajor = major;
    device.cudaComputeCapabilityMinor = minor;
    device.driverVersion = "550.54.14";
    return device;
}

ktt::db::DeviceInfo OpenClDevice()
{
    ktt::db::DeviceInfo device;
    device.name = "AMD Radeon RX 7900";
    device.vendor = "AMD";
    device.type = "GPU";
    device.computeApi = ktt::ComputeApi::OpenCL;
    device.extensions = "cl_khr_fp64";
    device.deviceIdentifier = "gfx1100";
    return device;
}

ktt::db::TuningInfo WithDevice(ktt::db::DeviceInfo device, const std::optional<std::string> &input = std::nullopt)
{
    auto info = MakeTuningInfo();
    info.device = std::move(device);
    info.inputData = input;
    return info;
}

// One source (222) with two spaces, five devices (two of them CUDA-compatible with an RTX 3090), runs with and without
// input data, saved in both formats and in three time units; a second source (999) with one run.
void SaveScenario(const Pair &pair)
{
    const auto cpu = MakeTuningInfo();
    auto cpuSmall = cpu;
    cpuSmall.inputData = "small";

    auto failed = MakeResult("cpu", 8, 1 * Millisecond);
    failed.SetStatus(ktt::ResultStatus::ComputationFailed);

    pair.Save(cpuSmall, {MakeResult("cpu", 64, 9 * Millisecond), MakeResult("cpu", 32, 11 * Millisecond), failed});

    {
        const ktt::db::TimeUnitGuard unit(ktt::TimeUnit::Microseconds);
        pair.Save(WithDevice(CudaDevice("NVIDIA GeForce RTX 3090", 8, 6), "large"),
            {MakeResult("cuda", 128, 2 * Millisecond + 500'000), MakeResult("cuda", 256, 6 * Millisecond)},
            ktt::OutputFormat::JSON_T4);
    }

    // A compatible CUDA device (8.7) without input data, saved with XML (stored as JSON remotely).
    pair.Save(WithDevice(CudaDevice("NVIDIA GeForce RTX 4090", 8, 7)), {MakeResult("cuda", 64, 4 * Millisecond)},
        ktt::OutputFormat::XML);

    // An incompatible CUDA device (7.5).
    {
        const ktt::db::TimeUnitGuard unit(ktt::TimeUnit::Seconds);
        pair.Save(WithDevice(CudaDevice("NVIDIA Tesla T4", 7, 5), "large"), {MakeResult("cuda", 32, 3 * Millisecond)});
    }

    pair.Save(WithDevice(OpenClDevice(), ""), {MakeResult("opencl", 16, 7 * Millisecond)});

    // A run without results still counts.
    pair.Save(cpu, {});

    // Another space of the same source.
    auto otherSpace = cpu;
    otherSpace.spaceInfo.spaceFingerprint = 444;
    pair.Save(otherSpace, {MakeResult("cpu", 64, 5 * Millisecond)});

    // Another source.
    auto otherSource = cpu;
    otherSource.spaceInfo.sourceFingerprint = 999;
    pair.Save(otherSource, {MakeResult("other", 64, 1 * Millisecond), MakeResult("other", 32, 2 * Millisecond)});
}

} // namespace

TEST_CASE("Remote and local SimpleGetBestResults agree", "[Database][Remote][Parity]")
{
    const Pair pair;
    SaveScenario(pair);

    const std::vector<ktt::db::TuningInfo> queries{MakeTuningInfo(),
        WithDevice(CudaDevice("NVIDIA GeForce RTX 3090", 8, 6)), WithDevice(CudaDevice("NVIDIA A100", 8, 0)),
        WithDevice(CudaDevice("NVIDIA Tesla T4", 7, 5)), WithDevice(OpenClDevice()),
        WithDevice(CudaDevice("Unknown GPU", 9, 0))};

    for (size_t i = 0; i < queries.size(); ++i)
    {
        for (const uint32_t limit : {0u, 1u, 2u, 50u})
        {
            CAPTURE(i, limit);
            const auto expected = Views(pair.local.SimpleGetBestResults(queries[i], limit));
            REQUIRE(Views(pair.remote.SimpleGetBestResults(queries[i], limit)) == expected);
        }
    }

    // An unknown source and an unknown space give nothing on both.
    auto unknown = MakeTuningInfo();
    unknown.spaceInfo.sourceFingerprint = 12345;
    REQUIRE(pair.remote.SimpleGetBestResults(unknown).empty());
    REQUIRE(pair.local.SimpleGetBestResults(unknown).empty());
    unknown = MakeTuningInfo();
    unknown.spaceInfo.parameterFingerprint = 12345;
    REQUIRE(pair.remote.SimpleGetBestResults(unknown).empty());
    REQUIRE(pair.local.SimpleGetBestResults(unknown).empty());

    // A sanity check that the scenario matches something: the 3090 sees itself and the 8.7 device.
    REQUIRE(Views(pair.remote.SimpleGetBestResults(queries[1])).size() == 3);
}

TEST_CASE("Remote and local GetBestResults agree", "[Database][Remote][Parity]")
{
    const Pair pair;
    SaveScenario(pair);

    const auto space = MakeTuningInfo().spaceInfo;
    const auto isCuda = [](const ktt::db::DeviceInfo &device) { return device.computeApi == ktt::ComputeApi::CUDA; };
    const auto isPlainDevice = [](const ktt::db::DeviceInfo &device)
    {
        // Compares the fields whose empty values the remote side maps the way the local one reads them.
        return !device.extensions.has_value() && !device.deviceIdentifier.has_value() && device.driverVersion.empty();
    };
    const auto isLarge = [](const std::string &input) { return input == "large"; };
    const auto any = [](const std::string &) { return true; };
    const auto isEmpty = [](const std::string &input) { return input.empty(); };

    const std::vector<GetBestResultsQuery> queries{
        {space, std::nullopt, std::nullopt, 50},
        {space, std::nullopt, std::nullopt, 1},
        {space, std::nullopt, std::nullopt, 3},
        {space, isCuda, std::nullopt, 50},
        {space, isPlainDevice, std::nullopt, 50},
        {space, std::nullopt, isLarge, 50},
        {space, std::nullopt, any, 50},
        {space, std::nullopt, isEmpty, 50},
        {space, isCuda, isLarge, 2},
        {space, [](const ktt::db::DeviceInfo &) { return false; }, std::nullopt, 50},
        {space, std::nullopt, std::nullopt, 0}};

    for (size_t i = 0; i < queries.size(); ++i)
    {
        CAPTURE(i);
        const auto expected = Views(pair.local.GetBestResults(queries[i]));
        REQUIRE(Views(pair.remote.GetBestResults(queries[i])) == expected);
    }

    // A sanity check: the whole space has 7 Ok results.
    REQUIRE(pair.remote.GetBestResults(queries[0]).size() == 7);

    auto unknown = space;
    unknown.spaceFingerprint = 12345;
    REQUIRE(pair.remote.GetBestResults({unknown, std::nullopt, std::nullopt, 50}).empty());
    REQUIRE(pair.local.GetBestResults({unknown, std::nullopt, std::nullopt, 50}).empty());
}

TEST_CASE("Remote and local GetBestResults agree across several pages of runs", "[Database][Remote][Parity]")
{
    const Pair pair;

    // 520 runs: more than one page of 500 on both sides. Each has a distinct duration, so there are no ties.
    for (uint64_t i = 0; i < 520; ++i)
    {
        auto info = MakeTuningInfo();
        info.inputData = "input " + std::to_string(i % 7);
        const auto duration = static_cast<ktt::Nanoseconds>((1000 - i) * Millisecond);
        pair.Save(info, {MakeResult("kernel", i, duration)});
    }

    REQUIRE(pair.server->RunCount() == 520);

    const auto space = MakeTuningInfo().spaceInfo;
    const auto isThree = [](const std::string &input) { return input == "input 3"; };

    for (const auto &query : std::vector<GetBestResultsQuery>{{space, std::nullopt, std::nullopt, 10},
             {space, std::nullopt, isThree, 1000}, {space, std::nullopt, std::nullopt, 1000}})
    {
        const auto expected = Views(pair.local.GetBestResults(query));
        REQUIRE(Views(pair.remote.GetBestResults(query)) == expected);
    }

    // The best result is in the second page of runs.
    REQUIRE(pair.remote.GetBestResults({space, std::nullopt, std::nullopt, 1}).at(0).GetKernelDuration()
        == 481 * Millisecond);
}

TEST_CASE("Remote and local GetStatsForSource agree, except for the name", "[Database][Remote][Parity]")
{
    const Pair pair;
    SaveScenario(pair);

    for (const size_t fingerprint : {size_t{222}, size_t{999}, size_t{12345}})
    {
        CAPTURE(fingerprint);
        const auto local = pair.local.GetStatsForSource(fingerprint);
        const auto remote = pair.remote.GetStatsForSource(fingerprint);

        REQUIRE(local.has_value() == remote.has_value());

        if (!local)
            continue;

        REQUIRE(remote->spaceCount == local->spaceCount);
        REQUIRE(remote->deviceCount == local->deviceCount);
        REQUIRE(remote->runCount == local->runCount);
        REQUIRE(remote->resultCount == local->resultCount);
        REQUIRE_FALSE(remote->name.has_value());
    }

    const auto stats = pair.remote.GetStatsForSource(222);
    REQUIRE(stats->spaceCount == 2);
    REQUIRE(stats->deviceCount == 5);
    REQUIRE(stats->runCount == 7);
    REQUIRE(stats->resultCount == 8);
}

TEST_CASE("Remote and local reject the same invalid input", "[Database][Remote][Parity]")
{
    const Pair pair;
    auto info = MakeTuningInfo();
    info.device.name.clear();

    for (const IDatabase *db : pair.Both())
    {
        REQUIRE_THROWS_AS(db->SaveResults(info, {MakeResult("kernel", 64, 1 * Millisecond)}), ktt::KttException);
        REQUIRE_THROWS_AS(db->SimpleGetBestResults(info), ktt::KttException);
    }

    REQUIRE(pair.server->RequestCount() == 0);
}

#endif // KTT_REMOTE_DATABASE && KTT_DATABASE
