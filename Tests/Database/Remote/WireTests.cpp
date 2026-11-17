#include <catch.hpp>

#if defined(KTT_REMOTE_DATABASE)

// The wire types of the ktt-db application API (Database/Remote/Utility/Wire). The request tests check the bodies
// RemoteDatabase sends end to end; these check the conversions on their own.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/Utility/Wire/Common.h>
#include <Database/Remote/Utility/Wire/Device.h>
#include <Database/Remote/Utility/Wire/Query.h>
#include <Database/Remote/Utility/Wire/Result.h>
#include <Database/Remote/Utility/Wire/Run.h>
#include <Database/Remote/Utility/Wire/Source.h>
#include <Database/Utility/TimeUnitGuard.h>
#include <Database/Utility/Uuid.h>

#include "../DatabaseTestHelpers.h"

using json = nlohmann::json;
using ktt::db::UuidGenerator;
using ktt::db::test::MakeResult;
using ktt::db::test::MakeTuningInfo;
using ktt::db::test::Millisecond;

namespace wire = ktt::db::wire;

TEST_CASE("Fingerprints are decimal strings on the wire", "[Database][Remote]")
{
    REQUIRE(wire::FingerprintToString(0) == "0");
    REQUIRE(wire::FingerprintToString(14695981039346656037ULL) == "14695981039346656037");

    for (const size_t value : {size_t{0}, size_t{1}, size_t{1099511628211ULL}, std::numeric_limits<size_t>::max()})
        REQUIRE(wire::FingerprintFromString(wire::FingerprintToString(value)) == value);
}

TEST_CASE("Invalid fingerprint strings are rejected", "[Database][Remote]")
{
    for (const std::string text : {"", "-1", "+1", " 1", "1 ", "12a", "0x10", "1.5", "18446744073709551616",
             "99999999999999999999999"})
    {
        CAPTURE(text);
        REQUIRE_THROWS_AS(wire::FingerprintFromString(text), ktt::KttException);
    }
}

TEST_CASE("A device is sent with null for what it does not have", "[Database][Remote]")
{
    ktt::db::DeviceInfo device = MakeTuningInfo().device;
    device.driverVersion = "";
    device.cudaComputeCapabilityMajor = 8;

    const json sent = wire::DeviceInfo::From(device);

    REQUIRE(sent == json{{"name", "Test CPU"}, {"vendor", "Test Vendor"}, {"type", "CPU"}, {"compute_api", "cpp"},
        {"device_identifier", nullptr}, {"driver_version", nullptr}, {"extensions", nullptr},
        {"cuda_compute_capability_major", 8}, {"cuda_compute_capability_minor", nullptr}});
}

TEST_CASE("A device is read as the local Database reads it", "[Database][Remote]")
{
    json received{{"name", "GPU"}, {"vendor", "NVIDIA"}, {"type", "GPU"}, {"compute_api", "cuda"},
        {"device_identifier", ""}, {"extensions", nullptr}, {"cuda_compute_capability_minor", 6}};

    const ktt::db::DeviceInfo device = received.get<wire::DeviceInfo>().ToDeviceInfo();

    REQUIRE(device.computeApi == ktt::ComputeApi::CUDA);
    REQUIRE_FALSE(device.deviceIdentifier.has_value());
    REQUIRE(device.driverVersion.empty());
    REQUIRE_FALSE(device.extensions.has_value());
    REQUIRE_FALSE(device.cudaComputeCapabilityMajor.has_value());
    REQUIRE(device.cudaComputeCapabilityMinor == 6u);

    received["compute_api"] = "metal";
    REQUIRE_THROWS_AS(received.get<wire::DeviceInfo>(), ktt::KttException);
}

TEST_CASE("A configuration keeps the JSON type of each value", "[Database][Remote]")
{
    const wire::Configuration configuration{{"a", int64_t{-3}}, {"b", uint64_t{4}}, {"c", 0.5}, {"d", true},
        {"e", std::string("x")}};

    const json sent = wire::ConfigurationToJson(configuration);
    REQUIRE(sent.dump() == R"({"a":-3,"b":4,"c":0.5,"d":true,"e":"x"})");
    REQUIRE(wire::ConfigurationFromJson(sent) == configuration);

    REQUIRE_THROWS_AS(wire::ConfigurationFromJson(json::array()), ktt::KttException);
    REQUIRE_THROWS_AS(wire::ConfigurationFromJson(json{{"a", nullptr}}), ktt::KttException);
}

TEST_CASE("A saved run sends its successful results in the current time unit", "[Database][Remote]")
{
    const ktt::db::TimeUnitGuard guard(ktt::TimeUnit::Microseconds);
    ktt::KernelResult failed = MakeResult("kernel", 32, 1 * Millisecond);
    failed.SetStatus(ktt::ResultStatus::ComputationFailed);

    const auto id = UuidGenerator::GenerateUuidV7();
    const json sent = wire::SaveRunRequest::From(id, MakeTuningInfo(),
        {MakeResult("kernel", 64, 2 * Millisecond), failed}, ktt::OutputFormat::XML);

    REQUIRE(sent.at("id") == UuidGenerator::ToString(id));
    REQUIRE(sent.at("output_format") == "json");
    REQUIRE(sent.at("time_unit") == "microseconds");
    REQUIRE(sent.at("tuning_info").at("space_info").at("source_fingerprint") == "222");
    REQUIRE(sent.at("results").size() == 1);

    const json &result = sent.at("results")[0];
    REQUIRE(result.at("duration") == 2 * Millisecond);
    REQUIRE(result.at("status") == "ok");
    REQUIRE(result.at("configuration") == json{{"block_size", 64}});
    REQUIRE(result.at("result").is_object());
}

TEST_CASE("A result is read back in the time unit of its run", "[Database][Remote]")
{
    json payload;
    {
        const ktt::db::TimeUnitGuard guard(ktt::TimeUnit::Microseconds);
        const auto request = wire::SaveRunRequest::From(UuidGenerator::GenerateUuidV7(), MakeTuningInfo(),
            {MakeResult("kernel", 64, 3 * Millisecond)}, ktt::OutputFormat::JSON);
        payload = request.results.at(0).result;
    }

    const json list{{"results",
        {{{"duration", 3 * Millisecond}, {"total_duration", 3 * Millisecond}, {"output_format", "json"},
            {"time_unit", "microseconds"}, {"result", payload}}}}};

    const auto results = list.get<wire::ResultList>().results;
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].duration == static_cast<int64_t>(3 * Millisecond));
    REQUIRE(results[0].ToKernelResult().GetKernelDuration() == 3 * Millisecond);

    REQUIRE_THROWS_AS((json{{"results", json::object()}}.get<wire::ResultList>()), ktt::KttException);
}

TEST_CASE("The result query selects only what ResultQueryItem holds", "[Database][Remote]")
{
    ktt::db::ResultQuery query;
    query.filter = "duration lt 5000";

    REQUIRE(json(wire::ResultQueryRequest::From(query)) == json{{"filter", "duration lt 5000"},
        {"orderby", "duration asc"}, {"top", 50}, {"skip", 0}, {"select", "run,device,source,result"}});

    query.includeResult = false;
    REQUIRE(json(wire::ResultQueryRequest::From(query)).at("select") == "run,device,source");
}

TEST_CASE("The save response says whether the server has the code, or nothing", "[Database][Remote]")
{
    REQUIRE(json{{"source_has_code", false}}.get<wire::SaveRunResponse>().sourceHasCode == false);
    REQUIRE_FALSE(json::object().get<wire::SaveRunResponse>().sourceHasCode.has_value());
    REQUIRE_FALSE(json{{"source_has_code", "no"}}.get<wire::SaveRunResponse>().sourceHasCode.has_value());
}

TEST_CASE("Source statistics must not be negative", "[Database][Remote]")
{
    json stats{{"space_count", 3}, {"device_count", 2}, {"run_count", 41}, {"result_count", 20992}};
    REQUIRE(stats.get<wire::SourceStatsResponse>().ToSourceStats().resultCount == 20992);

    stats["run_count"] = -1;
    REQUIRE_THROWS_AS(stats.get<wire::SourceStatsResponse>(), ktt::KttException);
}

#endif // KTT_REMOTE_DATABASE
