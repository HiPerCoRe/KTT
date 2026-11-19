#include <catch.hpp>

#if defined(KTT_REMOTE_DATABASE)

// R8.4: against the ktt-db server this build talks to (RemoteDatabase::GetHost, by default
// https://supercuda.fi.muni.cz:8443, which needs the FI MUNI VPN; another one is set with the premake option
// --remote-database-host). Hidden (run it with: ./Tests "[integration]") and skipped unless KTT_DB_API_KEY is set. The
// key needs the write permission. KTT_DB_CA_BUNDLE optionally names a CA bundle (PEM) for a server with a self-signed
// certificate, such as the local development backend. Every run uses new random fingerprints, so it can be repeated
// against the same server.

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <random>
#include <string>

#include <Api/KttException.h>
#include <Database/Remote/RemoteDatabase.h>
#include <Database/Utility/TimeUnitGuard.h>

#include "../DatabaseTestHelpers.h"

using ktt::db::test::MakeResult;
using ktt::db::test::MakeTuningInfo;
using ktt::db::test::Millisecond;

TEST_CASE("RemoteDatabase saves and reads results on a ktt-db server", "[.integration][Database][Remote]")
{
    const char *apiKey = std::getenv("KTT_DB_API_KEY");

    if (apiKey == nullptr || *apiKey == '\0')
    {
        WARN("KTT_DB_API_KEY is not set: skipped");
        return;
    }

    ktt::db::RemoteDatabaseOptions options;

    if (const char *caBundle = std::getenv("KTT_DB_CA_BUNDLE"); caBundle != nullptr && *caBundle != '\0')
        options.caBundle = caBundle;

    const ktt::db::RemoteDatabase db(apiKey, options);

    std::random_device random;
    std::uniform_int_distribution<size_t> fingerprints(1, SIZE_MAX);
    auto info = MakeTuningInfo();
    info.spaceInfo.sourceFingerprint = fingerprints(random);
    info.spaceInfo.parameterFingerprint = fingerprints(random);
    info.spaceInfo.spaceFingerprint = fingerprints(random);
    info.spaceInfo.sourceName = "ktt integration test";
    info.inputData = "integration";

    // The first request doubles as a reachability check: a transport failure (no connection, DNS, TLS, timeout) gets a
    // hint instead of the bare libcurl message.
    std::optional<ktt::db::SourceStats> initialStats;

    try
    {
        initialStats = db.GetStatsForSource(info.spaceInfo.sourceFingerprint);
    }
    catch (const ktt::KttException &exception)
    {
        const std::string message = exception.what();

        // RemoteClient reports a transport failure as "ktt-db request <method> <path> failed...", which the
        // operation's name then prefixes; an HTTP error status has no such part.
        if (message.find("ktt-db request ") != std::string::npos)
        {
            FAIL("Cannot reach the ktt-db server " << ktt::db::RemoteDatabase::GetHost() << ": " << message
                << "\nYou must be connected to the FI MUNI VPN, or the server is down :D");
        }

        throw;
    }

    REQUIRE_FALSE(initialStats.has_value());
    REQUIRE(db.SimpleGetBestResults(info).empty());

    {
        const ktt::db::TimeUnitGuard unit(ktt::TimeUnit::Microseconds);
        db.SaveResults(info, {MakeResult("kernel", 64, 3 * Millisecond), MakeResult("kernel", 32, 2 * Millisecond)});
    }

    db.SaveResults(info, {MakeResult("kernel", 128, 1 * Millisecond)}, {ktt::OutputFormat::JSON_T4, 2});

    const auto best = db.SimpleGetBestResults(info, 2);
    REQUIRE(best.size() == 2);
    REQUIRE(best[0].GetKernelDuration() == 1 * Millisecond);
    REQUIRE(best[1].GetKernelDuration() == 2 * Millisecond);

    const auto filtered = db.GetBestResults({info.spaceInfo, std::nullopt,
        [](const std::string &input) { return input == "integration"; }, 10});
    REQUIRE(filtered.size() == 3);
    REQUIRE(filtered[2].GetKernelDuration() == 3 * Millisecond);

    const auto stats = db.GetStatsForSource(info.spaceInfo.sourceFingerprint);
    REQUIRE(stats.has_value());
    REQUIRE(stats->spaceCount == 1);
    REQUIRE(stats->deviceCount == 1);
    REQUIRE(stats->runCount == 2);
    REQUIRE(stats->resultCount == 3);
}

#endif // KTT_REMOTE_DATABASE
