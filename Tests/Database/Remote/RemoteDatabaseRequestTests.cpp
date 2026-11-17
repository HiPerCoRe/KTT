#include <catch.hpp>

#if defined(KTT_REMOTE_DATABASE)

// R8.1: the requests RemoteDatabase sends and how it maps the responses, against a scripted fake HttpClient.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/RemoteDatabase.h>
#include <Database/Utility/ResultSerialization.h>
#include <Database/Utility/TimeUnitGuard.h>
#include <Database/Utility/Uuid.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>

#include "../DatabaseTestHelpers.h"
#include "FakeHttpClient.h"

using json = nlohmann::json;
using ktt::db::GetBestResultsQuery;
using ktt::db::RemoteDatabase;
using ktt::db::RemoteDatabaseOptions;
using ktt::db::UuidGenerator;
using ktt::db::test::FakeHttpClient;
using ktt::db::test::MakeResult;
using ktt::db::test::MakeTuningInfo;
using ktt::db::test::Millisecond;
using ktt::db::test::RecordedRequest;

namespace
{

const std::string ApiKey = "kttak-secret-test-key";
const std::string BaseUrl = RemoteDatabase::GetHost();
const std::string AppUrl = BaseUrl + "/ktt-db/app";

const std::string SaveRunResponse = R"({"run_id":"0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b17","source_id":1,"space_id":2,)"
    R"("result_count":2,"realm_ids":["00000000-0000-4000-8000-000000000001"],"source_has_code":true})";

struct Fixture
{
    std::shared_ptr<FakeHttpClient> http = std::make_shared<FakeHttpClient>();

    // No retries unless a test asks for them: a retry sleeps for real (250 ms), because the public constructor takes
    // no sleeper.
    RemoteDatabase MakeDatabase(const uint32_t maxRetries = 0) const
    {
        RemoteDatabaseOptions options;
        options.maxRetries = maxRetries;
        return RemoteDatabase(ApiKey, options, http);
    }

    std::vector<RecordedRequest> Requests() const
    {
        return http->Requests();
    }
};

json Body(const RecordedRequest &request)
{
    return json::parse(request.body);
}

// A ResultOut whose payload is serialized in the given format and time unit, as a run saved with that unit holds it.
json ResultOut(const ktt::KernelResult &result, const ktt::OutputFormat format, const ktt::TimeUnit unit)
{
    const ktt::db::TimeUnitGuard guard(unit);
    return json{{"duration", result.GetKernelDuration()}, {"total_duration", result.GetTotalDuration()},
        {"output_format", format == ktt::OutputFormat::JSON_T4 ? "json_t4" : "json"},
        {"time_unit", ktt::db::TimeUnitToName(unit)},
        {"result", json::parse(ktt::db::ResultSerialization::SerializeResult(result, format, 0))}};
}

json ResultList(const std::vector<ktt::Nanoseconds> &durations)
{
    json results = json::array();

    for (const auto duration : durations)
        results.push_back(ResultOut(MakeResult("kernel", 64, duration), ktt::OutputFormat::JSON,
            ktt::TimeUnit::Milliseconds));

    return json{{"results", results}};
}

json DeviceJson(const std::string &name = "Test CPU")
{
    return json{{"name", name}, {"vendor", "Test Vendor"}, {"type", "CPU"}, {"compute_api", "cpp"},
        {"device_identifier", nullptr}, {"driver_version", nullptr}, {"extensions", nullptr},
        {"cuda_compute_capability_major", nullptr}, {"cuda_compute_capability_minor", nullptr}};
}

json RunJson(const std::string &id, const std::optional<std::string> &inputData = std::nullopt,
    const json &device = DeviceJson())
{
    return json{{"run_id", id}, {"input_data", inputData ? json(*inputData) : json(nullptr)}, {"device", device},
        {"tuner", {{"name", "KTT"}, {"version", "2.3.1"}}}, {"output_format", "json"},
        {"time_unit", "milliseconds"}, {"created_at", "2026-11-06T10:00:00Z"}};
}

std::vector<std::string> RunIds(const size_t count)
{
    std::vector<std::string> ids;

    for (size_t i = 0; i < count; ++i)
        ids.push_back(UuidGenerator::ToString(UuidGenerator::GenerateUuidV7()));

    return ids;
}

std::string RunPage(const std::vector<std::string> &ids)
{
    json runs = json::array();

    for (const auto &id : ids)
        runs.push_back(RunJson(id));

    return json{{"runs", runs}}.dump();
}

const std::string SpaceFound = R"({"space_id":31,"source_id":14})";
const std::string SpacePath = AppUrl + "/source/222/space/31";

std::vector<ktt::Nanoseconds> Durations(const std::vector<ktt::KernelResult> &results)
{
    std::vector<ktt::Nanoseconds> durations;

    for (const auto &result : results)
        durations.push_back(result.GetKernelDuration());

    return durations;
}

void RequireCommonHeaders(const RecordedRequest &request, const bool hasBody)
{
    REQUIRE(request.Header("x-api-key") == ApiKey);
    REQUIRE(request.Header("correlation-id").has_value());
    REQUIRE(UuidGenerator::GetVersion(UuidGenerator::Parse(*request.Header("correlation-id"))) == 4);
    REQUIRE(request.Header("content-type") == (hasBody ? std::optional<std::string>("application/json") : std::nullopt));
}

std::string ExceptionText(const std::function<void()> &function)
{
    try
    {
        function();
    }
    catch (const ktt::KttException &error)
    {
        return error.what();
    }

    FAIL("no KttException thrown");
    return {};
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// SaveResults (R4.1)

TEST_CASE("SaveResults posts the run with a UUIDv7 id and the documented body", "[Database][Remote]")
{
    Fixture fixture;
    fixture.http->EnqueueResponse(201, SaveRunResponse);
    const auto db = fixture.MakeDatabase();
    const ktt::db::TimeUnitGuard unit(ktt::TimeUnit::Microseconds);

    auto failed = MakeResult("kernel", 32, 1 * Millisecond);
    failed.SetStatus(ktt::ResultStatus::ComputationFailed);

    const ktt::KernelConfiguration configuration({ktt::ParameterPair("int", int64_t{-3}),
        ktt::ParameterPair("unsigned", uint64_t{18446744073709551615ULL}), ktt::ParameterPair("double", 0.5),
        ktt::ParameterPair("bool", true), ktt::ParameterPair("string", std::string("abc")),
        ktt::ParameterPair("-DFAST", true, true)});
    ktt::ComputationResult computation("kernelFunction");
    computation.SetDurationData(4 * Millisecond, 0, 0);
    ktt::KernelResult typed("kernel", configuration, {computation}, "2026-09-09T00:00:00");

    db.SaveResults(MakeTuningInfo(), {MakeResult("kernel", 64, 3 * Millisecond), failed, typed});

    const auto requests = fixture.Requests();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests[0].method == "POST");
    REQUIRE(requests[0].url == AppUrl + "/run");
    RequireCommonHeaders(requests[0], true);

    const json body = Body(requests[0]);
    REQUIRE(UuidGenerator::GetVersion(UuidGenerator::Parse(body.at("id").get<std::string>())) == 7);
    REQUIRE(UuidGenerator::ToString(UuidGenerator::Parse(body.at("id").get<std::string>())) == body.at("id"));
    REQUIRE(body.at("output_format") == "json");
    REQUIRE(body.at("time_unit") == "microseconds");

    const json &info = body.at("tuning_info");
    REQUIRE(info.at("space_info") == json{{"source_fingerprint", "222"}, {"parameter_fingerprint", "111"},
        {"space_fingerprint", "333"}, {"source_name", nullptr}, {"space_name", nullptr}});
    REQUIRE(info.at("device") == DeviceJson());
    REQUIRE(info.at("tuner").at("name") == "KTT");
    REQUIRE(info.at("tuner").at("version").is_string());
    REQUIRE(info.at("input_data").is_null());

    // The failed result is not sent.
    const json &results = body.at("results");
    REQUIRE(results.size() == 2);
    REQUIRE(results[0].at("duration") == 3 * Millisecond);
    REQUIRE(results[0].at("total_duration") == 3 * Millisecond);
    REQUIRE(results[0].at("status") == "ok");
    REQUIRE(results[0].at("configuration") == json{{"block_size", 64}});
    REQUIRE(results[0].at("result").is_object());
    REQUIRE(results[0].at("result").at("KernelName") == "kernel");

    const json &typedConfiguration = results[1].at("configuration");
    REQUIRE(typedConfiguration.at("int").is_number_integer());
    REQUIRE(typedConfiguration.at("int") == -3);
    REQUIRE(typedConfiguration.at("unsigned").is_number_unsigned());
    REQUIRE(typedConfiguration.at("unsigned").get<uint64_t>() == 18446744073709551615ULL);
    REQUIRE(typedConfiguration.at("double").is_number_float());
    REQUIRE(typedConfiguration.at("bool") == true);
    REQUIRE(typedConfiguration.at("string") == "abc");
    REQUIRE(typedConfiguration.at("-DFAST").is_string());
}

TEST_CASE("SaveResults sends the names, the input data and a full device", "[Database][Remote]")
{
    Fixture fixture;
    fixture.http->EnqueueResponse(200, SaveRunResponse);
    const auto db = fixture.MakeDatabase();

    auto info = MakeTuningInfo();
    info.spaceInfo.sourceName = "coulomb_sum_3d";
    info.spaceInfo.spaceName = "default";
    info.spaceInfo.sourceFingerprint = 14695981039346656037ULL;
    info.inputData = "atoms=4096";
    info.device.name = "NVIDIA GeForce RTX 3090";
    info.device.vendor = "NVIDIA";
    info.device.type = "GPU";
    info.device.computeApi = ktt::ComputeApi::CUDA;
    info.device.cudaComputeCapabilityMajor = 8;
    info.device.cudaComputeCapabilityMinor = 6;
    info.device.driverVersion = "550.54.14";
    info.device.deviceIdentifier = "GPU-6c3f1a2b";

    // 200 (the run already existed) is a success too.
    db.SaveResults(info, {});

    const json body = Body(fixture.Requests().at(0));
    const json &spaceInfo = body.at("tuning_info").at("space_info");
    REQUIRE(spaceInfo.at("source_fingerprint") == "14695981039346656037");
    REQUIRE(spaceInfo.at("source_name") == "coulomb_sum_3d");
    REQUIRE(spaceInfo.at("space_name") == "default");
    REQUIRE(body.at("tuning_info").at("input_data") == "atoms=4096");
    REQUIRE(body.at("tuning_info").at("device") == json{{"name", "NVIDIA GeForce RTX 3090"}, {"vendor", "NVIDIA"},
        {"type", "GPU"}, {"compute_api", "cuda"}, {"device_identifier", "GPU-6c3f1a2b"},
        {"driver_version", "550.54.14"}, {"extensions", nullptr}, {"cuda_compute_capability_major", 8},
        {"cuda_compute_capability_minor", 6}});

    // An empty result list still creates a run.
    REQUIRE(body.at("results") == json::array());
}

TEST_CASE("SaveResults sends XML as json and JSON_T4 as json_t4", "[Database][Remote]")
{
    Fixture fixture;
    fixture.http->EnqueueResponse(201, SaveRunResponse);
    fixture.http->EnqueueResponse(201, SaveRunResponse);
    const auto db = fixture.MakeDatabase();
    const auto results = std::vector<ktt::KernelResult>{MakeResult("kernel", 64, 3 * Millisecond)};

    db.SaveResults(MakeTuningInfo(), results, {ktt::OutputFormat::XML, 4});
    db.SaveResults(MakeTuningInfo(), results, {ktt::OutputFormat::JSON_T4, 2});

    const auto requests = fixture.Requests();
    const json xml = Body(requests.at(0));
    REQUIRE(xml.at("output_format") == "json");
    REQUIRE(xml.at("results")[0].at("result").at("KernelName") == "kernel");

    const json t4 = Body(requests.at(1));
    REQUIRE(t4.at("output_format") == "json_t4");
    REQUIRE(t4.at("results")[0].at("result").is_object());

    // Every save gets its own id.
    REQUIRE(xml.at("id") != t4.at("id"));
}

TEST_CASE("SaveResults validates the TuningInfo before any request", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    auto info = MakeTuningInfo();
    info.spaceInfo.sourceFingerprint = 0;

    REQUIRE_THROWS_AS(db.SaveResults(info, {}), ktt::KttException);
    REQUIRE_THROWS_AS(db.SimpleGetBestResults(info), ktt::KttException);
    REQUIRE(fixture.Requests().empty());
}

TEST_CASE("SaveResults resends the same run id after a timeout and a 503", "[Database][Remote]")
{
    Fixture fixture;
    fixture.http->EnqueueTransportError("Timeout was reached", true);
    fixture.http->EnqueueResponse(503, R"({"error":"concurrent save"})");
    fixture.http->EnqueueResponse(201, SaveRunResponse);
    const auto db = fixture.MakeDatabase(2);

    db.SaveResults(MakeTuningInfo(), {MakeResult("kernel", 64, 3 * Millisecond)});

    const auto requests = fixture.Requests();
    REQUIRE(requests.size() == 3);
    REQUIRE(requests[0].body == requests[1].body);
    REQUIRE(requests[1].body == requests[2].body);
    REQUIRE(requests[0].Header("correlation-id") == requests[2].Header("correlation-id"));
}

TEST_CASE("SaveResults maps error responses to exceptions without retrying them", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase(3);
    const auto save = [&db] { db.SaveResults(MakeTuningInfo(), {}); };

    SECTION("400 is not retried")
    {
        fixture.http->EnqueueResponse(400, R"({"error":"tuning_info.device.name must not be empty"})");
        const std::string text = ExceptionText(save);
        REQUIRE(text.find("RemoteDatabase::SaveResults") != std::string::npos);
        REQUIRE(text.find("HTTP 400") != std::string::npos);
        REQUIRE(text.find("tuning_info.device.name must not be empty") != std::string::npos);
        REQUIRE(fixture.Requests().size() == 1);
    }

    SECTION("409 throws")
    {
        fixture.http->EnqueueResponse(409, R"({"error":"run exists"})");
        const std::string text = ExceptionText(save);
        REQUIRE(text.find("HTTP 409") != std::string::npos);
        REQUIRE(text.find("run exists") != std::string::npos);
    }

    SECTION("403 and 413 throw")
    {
        fixture.http->EnqueueResponse(403, R"({"error":"read-only key"})");
        fixture.http->EnqueueResponse(413, "");
        REQUIRE(ExceptionText(save).find("read-only key") != std::string::npos);
        REQUIRE(ExceptionText(save).find("HTTP 413") != std::string::npos);
    }

    SECTION("a success body that is not JSON throws")
    {
        fixture.http->EnqueueResponse(201, "<html>proxy</html>");
        REQUIRE(ExceptionText(save).find("not a JSON object") != std::string::npos);
    }

    SECTION("an unexpected 2xx throws")
    {
        fixture.http->EnqueueResponse(204, "");
        REQUIRE(ExceptionText(save).find("HTTP 204") != std::string::npos);
    }
}

TEST_CASE("The API key never appears in exception messages", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();

    fixture.http->EnqueueResponse(401, R"({"error":"invalid api key )" + ApiKey + R"("})");
    fixture.http->EnqueueResponse(500, "upstream said: " + ApiKey);
    fixture.http->EnqueueTransportError("connection reset while sending " + ApiKey, false);
    fixture.http->EnqueueResponse(200, "not json " + ApiKey);

    const std::vector<std::string> texts{
        ExceptionText([&db] { db.SaveResults(MakeTuningInfo(), {}); }),
        ExceptionText([&db] { db.SimpleGetBestResults(MakeTuningInfo()); }),
        ExceptionText([&db] { db.GetStatsForSource(222); }),
        ExceptionText([&db] { db.GetStatsForSource(222); })};

    for (const auto &text : texts)
    {
        CAPTURE(text);
        REQUIRE(text.find(ApiKey) == std::string::npos);
        REQUIRE(text.find("<api key>") != std::string::npos);
    }

    REQUIRE(texts[0].find("HTTP 401") != std::string::npos);
    REQUIRE(texts[1].find("HTTP 500") != std::string::npos);
    REQUIRE(texts[2].find("RemoteDatabase::GetStatsForSource") != std::string::npos);
}

// ---------------------------------------------------------------------------------------------------------------------
// SimpleGetBestResults (R4.2)

TEST_CASE("SimpleGetBestResults posts the tuning info and the limit", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();

    // Payloads in two time units and both formats: each is read in its own run's unit, and the server order is kept.
    const ktt::db::TimeUnitGuard active(ktt::TimeUnit::Milliseconds);
    json list{{"results",
        {ResultOut(MakeResult("a", 64, 2 * Millisecond), ktt::OutputFormat::JSON, ktt::TimeUnit::Microseconds),
            ResultOut(MakeResult("b", 32, 3 * Millisecond), ktt::OutputFormat::JSON_T4, ktt::TimeUnit::Nanoseconds),
            ResultOut(MakeResult("c", 16, 5 * Millisecond), ktt::OutputFormat::JSON, ktt::TimeUnit::Seconds)}}};
    fixture.http->EnqueueResponse(200, list.dump());

    auto info = MakeTuningInfo();
    info.spaceInfo.sourceName = "ignored by the server";
    const auto results = db.SimpleGetBestResults(info, 7);

    const auto requests = fixture.Requests();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests[0].method == "POST");
    REQUIRE(requests[0].url == AppUrl + "/result/best");
    RequireCommonHeaders(requests[0], true);

    const json body = Body(requests[0]);
    REQUIRE(body.at("limit") == 7);
    REQUIRE(body.at("tuning_info").at("space_info").at("space_fingerprint") == "333");
    REQUIRE(body.at("tuning_info").at("device") == DeviceJson());

    REQUIRE(Durations(results) == std::vector<ktt::Nanoseconds>{2 * Millisecond, 3 * Millisecond, 5 * Millisecond});
    REQUIRE(results[0].GetKernelName() == "a");
    REQUIRE(results[2].GetKernelName() == "c"); // JSON T4 keeps no kernel name, so "b" is not checked

    // The active unit is restored after reading.
    REQUIRE(ktt::TimeConfiguration::GetInstance().GetTimeUnit() == ktt::TimeUnit::Milliseconds);
}

TEST_CASE("SimpleGetBestResults handles the limits on the client", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();

    REQUIRE(db.SimpleGetBestResults(MakeTuningInfo(), 0).empty());
    REQUIRE_THROWS_AS(db.SimpleGetBestResults(MakeTuningInfo(), 1001), ktt::KttException);
    REQUIRE(fixture.Requests().empty());

    fixture.http->EnqueueResponse(200, R"({"results":[]})");
    REQUIRE(db.SimpleGetBestResults(MakeTuningInfo(), 1000).empty());
    REQUIRE(Body(fixture.Requests().at(0)).at("limit") == 1000);

    // The default limit of the interface.
    fixture.http->EnqueueResponse(200, R"({"results":[]})");
    const ktt::db::IDatabase &base = db;
    REQUIRE(base.SimpleGetBestResults(MakeTuningInfo()).empty());
    REQUIRE(Body(fixture.Requests().at(1)).at("limit") == 50);
}

TEST_CASE("SimpleGetBestResults maps error responses to exceptions", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase(3);

    // The endpoint answers an unknown source or space with an empty list, so a 404 means a wrong URL.
    fixture.http->EnqueueResponse(200, R"({"results":[]})");
    REQUIRE(db.SimpleGetBestResults(MakeTuningInfo()).empty());
    fixture.http->EnqueueResponse(404, R"({"error":"not found"})");
    REQUIRE(ExceptionText([&db] { db.SimpleGetBestResults(MakeTuningInfo()); }).find("HTTP 404")
        != std::string::npos);
    fixture.http->EnqueueResponse(404, "");
    REQUIRE_THROWS_AS(db.SimpleGetBestResults(MakeTuningInfo()), ktt::KttException);

    // A query timeout is not retried.
    fixture.http->EnqueueResponse(503, R"({"error":"the query took too long","code":"query_timeout"})");
    const std::string text = ExceptionText([&db] { db.SimpleGetBestResults(MakeTuningInfo()); });
    REQUIRE(text.find("HTTP 503") != std::string::npos);
    REQUIRE(text.find("query_timeout") != std::string::npos);
    REQUIRE(fixture.Requests().size() == 4);

    // A malformed success body throws a KttException, not a JSON exception.
    fixture.http->EnqueueResponse(200, R"({"results":[{"duration":"x"}]})");
    REQUIRE_THROWS_AS(db.SimpleGetBestResults(MakeTuningInfo()), ktt::KttException);
    fixture.http->EnqueueResponse(200, R"({"rows":[]})");
    REQUIRE_THROWS_AS(db.SimpleGetBestResults(MakeTuningInfo()), ktt::KttException);
}

// ---------------------------------------------------------------------------------------------------------------------
// GetBestResults (R4.3)

TEST_CASE("GetBestResults returns nothing for a limit of 0 and rejects a limit above 1000", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const auto info = MakeTuningInfo();

    REQUIRE(db.GetBestResults(GetBestResultsQuery{info.spaceInfo, std::nullopt, std::nullopt, 0}).empty());
    REQUIRE_THROWS_AS(db.GetBestResults(GetBestResultsQuery{info.spaceInfo, std::nullopt, std::nullopt, 1001}),
        ktt::KttException);
    REQUIRE(fixture.Requests().empty());
}

TEST_CASE("GetBestResults returns nothing for an unknown space", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    auto info = MakeTuningInfo();
    info.spaceInfo.sourceFingerprint = 14695981039346656037ULL;

    fixture.http->EnqueueResponse(404, R"({"error":"space not found"})");
    REQUIRE(db.GetBestResults(GetBestResultsQuery{info.spaceInfo, std::nullopt, std::nullopt, 10}).empty());

    const auto requests = fixture.Requests();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests[0].method == "GET");
    REQUIRE(requests[0].url
        == AppUrl + "/source/14695981039346656037/space?parameter_fingerprint=111&space_fingerprint=333");
    RequireCommonHeaders(requests[0], false);
    REQUIRE(requests[0].body.empty());
}

TEST_CASE("A 404 is not found only with a ktt-db error body", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const auto info = MakeTuningInfo();
    const GetBestResultsQuery query{info.spaceInfo, std::nullopt, std::nullopt, 10};

    // An unmatched route has an empty body; a proxy sends its own page; neither is "not found".
    for (const std::string &body : {std::string(), std::string("<html><body>404 Not Found</body></html>"),
        std::string(R"({"message":"not found"})")})
    {
        fixture.http->EnqueueResponse(404, body);
        REQUIRE(ExceptionText([&db, &query] { db.GetBestResults(query); }).find("HTTP 404") != std::string::npos);

        fixture.http->EnqueueResponse(200, SpaceFound);
        fixture.http->EnqueueResponse(404, body);
        REQUIRE(ExceptionText([&db, &query] { db.GetBestResults(query); }).find("HTTP 404") != std::string::npos);

        fixture.http->EnqueueResponse(200, SpaceFound);
        fixture.http->EnqueueResponse(200, RunPage(RunIds(1)));
        fixture.http->EnqueueResponse(404, body);
        REQUIRE(ExceptionText([&db, &query] { db.GetBestResults(query); }).find("HTTP 404") != std::string::npos);

        fixture.http->EnqueueResponse(404, body);
        REQUIRE(ExceptionText([&db] { db.GetStatsForSource(222); }).find("HTTP 404") != std::string::npos);
    }

    fixture.http->EnqueueResponse(404, R"({"error":"source not found"})");
    REQUIRE_FALSE(db.GetStatsForSource(222).has_value());
}

TEST_CASE("GetBestResults pages through the runs", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const auto info = MakeTuningInfo();
    const GetBestResultsQuery query{info.spaceInfo, std::nullopt, std::nullopt, 10};
    fixture.http->EnqueueResponse(200, SpaceFound);

    SECTION("0 runs")
    {
        fixture.http->EnqueueResponse(200, RunPage({}));
        REQUIRE(db.GetBestResults(query).empty());

        const auto requests = fixture.Requests();
        REQUIRE(requests.size() == 2);
        REQUIRE(requests[1].method == "GET");
        REQUIRE(requests[1].url == SpacePath + "/run?offset=0&limit=500");
    }

    SECTION("1 run")
    {
        const auto ids = RunIds(1);
        fixture.http->EnqueueResponse(200, RunPage(ids));
        fixture.http->EnqueueResponse(200, ResultList({2 * Millisecond, 4 * Millisecond}).dump());

        REQUIRE(Durations(db.GetBestResults(query)) == std::vector<ktt::Nanoseconds>{2 * Millisecond, 4 * Millisecond});

        const auto requests = fixture.Requests();
        REQUIRE(requests.size() == 3);
        REQUIRE(requests[2].method == "POST");
        REQUIRE(requests[2].url == SpacePath + "/result");
        RequireCommonHeaders(requests[2], true);
        REQUIRE(Body(requests[2]) == json{{"run_ids", ids}, {"limit", 10}});
    }

    SECTION("500 runs")
    {
        const auto ids = RunIds(500);
        fixture.http->EnqueueResponse(200, RunPage(ids));
        fixture.http->EnqueueResponse(200, ResultList({1 * Millisecond}).dump());
        fixture.http->EnqueueResponse(200, RunPage({}));

        REQUIRE(db.GetBestResults(query).size() == 1);

        const auto requests = fixture.Requests();
        REQUIRE(requests.size() == 4);
        REQUIRE(Body(requests[2]).at("run_ids").size() == 500);
        REQUIRE(requests[3].url == SpacePath + "/run?offset=500&limit=500");
    }

    SECTION("501 runs")
    {
        const auto ids = RunIds(501);
        fixture.http->EnqueueResponse(200, RunPage({ids.begin(), ids.begin() + 500}));
        fixture.http->EnqueueResponse(200, ResultList({1 * Millisecond, 3 * Millisecond}).dump());
        fixture.http->EnqueueResponse(200, RunPage({ids[500]}));
        fixture.http->EnqueueResponse(200, ResultList({2 * Millisecond}).dump());

        REQUIRE(Durations(db.GetBestResults(query))
            == std::vector<ktt::Nanoseconds>{1 * Millisecond, 2 * Millisecond, 3 * Millisecond});

        const auto requests = fixture.Requests();
        REQUIRE(requests.size() == 5);
        REQUIRE(requests[3].url == SpacePath + "/run?offset=500&limit=500");
        REQUIRE(Body(requests[4]) == json{{"run_ids", {ids[500]}}, {"limit", 10}});
    }

    SECTION("a run repeated on a later page is used once")
    {
        const auto ids = RunIds(502);
        fixture.http->EnqueueResponse(200, RunPage({ids.begin(), ids.begin() + 500}));
        fixture.http->EnqueueResponse(200, ResultList({1 * Millisecond}).dump());
        // A run saved meanwhile shifted the list by one, so the last run of page 1 comes again.
        fixture.http->EnqueueResponse(200, RunPage({ids[499], ids[500], ids[501]}));
        fixture.http->EnqueueResponse(200, ResultList({2 * Millisecond}).dump());

        REQUIRE(db.GetBestResults(query).size() == 2);

        const auto requests = fixture.Requests();
        REQUIRE(requests.size() == 5);
        REQUIRE(Body(requests[4]).at("run_ids") == json{ids[500], ids[501]});
    }

    SECTION("a 404 while paging returns nothing")
    {
        fixture.http->EnqueueResponse(200, RunPage(RunIds(1)));
        fixture.http->EnqueueResponse(404, R"({"error":"space not found"})");
        REQUIRE(db.GetBestResults(query).empty());
    }

    SECTION("an error while paging throws")
    {
        fixture.http->EnqueueResponse(400, R"({"error":"invalid offset"})");
        const std::string text = ExceptionText([&db, &query] { db.GetBestResults(query); });
        REQUIRE(text.find("RemoteDatabase::GetBestResults") != std::string::npos);
        REQUIRE(text.find("invalid offset") != std::string::npos);
    }
}

TEST_CASE("GetBestResults applies the predicates on the client", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const auto info = MakeTuningInfo();
    const auto ids = RunIds(4);

    json cuda = DeviceJson("GPU");
    cuda["compute_api"] = "cuda";
    cuda["cuda_compute_capability_major"] = 8;
    cuda["cuda_compute_capability_minor"] = 6;
    cuda["driver_version"] = "550.54.14";
    cuda["device_identifier"] = "";

    json openCl = DeviceJson("OpenCL CPU");
    openCl["compute_api"] = "opencl";
    openCl["extensions"] = "";

    const std::string page = json{{"runs",
        {RunJson(ids[0], "small"), RunJson(ids[1], std::nullopt, cuda), RunJson(ids[2], "large", openCl),
            RunJson(ids[3], "")}}}.dump();

    SECTION("predicates that drop everything send no result request")
    {
        fixture.http->EnqueueResponse(200, SpaceFound);
        fixture.http->EnqueueResponse(200, page);

        const GetBestResultsQuery query{info.spaceInfo, [](const ktt::db::DeviceInfo &) { return false; },
            std::nullopt, 10};
        REQUIRE(db.GetBestResults(query).empty());
        REQUIRE(fixture.Requests().size() == 2);
    }

    SECTION("the device predicate sees the device as the local Database reads it")
    {
        fixture.http->EnqueueResponse(200, SpaceFound);
        fixture.http->EnqueueResponse(200, page);
        fixture.http->EnqueueResponse(200, ResultList({1 * Millisecond}).dump());

        std::vector<ktt::db::DeviceInfo> seen;
        const GetBestResultsQuery query{info.spaceInfo, [&seen](const ktt::db::DeviceInfo &device)
        {
            seen.push_back(device);
            return device.computeApi == ktt::ComputeApi::CUDA;
        }, std::nullopt, 10};
        REQUIRE(db.GetBestResults(query).size() == 1);

        REQUIRE(seen.size() == 4);
        REQUIRE(seen[0].name == "Test CPU");
        REQUIRE(seen[0].computeApi == ktt::ComputeApi::Cpp);
        REQUIRE(seen[0].driverVersion.empty());
        REQUIRE_FALSE(seen[0].cudaComputeCapabilityMajor.has_value());
        REQUIRE(seen[1].cudaComputeCapabilityMajor == 8u);
        REQUIRE(seen[1].cudaComputeCapabilityMinor == 6u);
        REQUIRE(seen[1].driverVersion == "550.54.14");
        REQUIRE_FALSE(seen[1].deviceIdentifier.has_value());
        REQUIRE(seen[2].computeApi == ktt::ComputeApi::OpenCL);
        REQUIRE_FALSE(seen[2].extensions.has_value());

        REQUIRE(Body(fixture.Requests().at(2)).at("run_ids") == json{ids[1]});
    }

    SECTION("a run without input data is left out by an input predicate, as locally")
    {
        fixture.http->EnqueueResponse(200, SpaceFound);
        fixture.http->EnqueueResponse(200, page);
        fixture.http->EnqueueResponse(200, ResultList({1 * Millisecond}).dump());

        std::vector<std::string> seen;
        const GetBestResultsQuery query{info.spaceInfo, std::nullopt, [&seen](const std::string &input)
        {
            seen.push_back(input);
            return true;
        }, 10};
        db.GetBestResults(query);

        REQUIRE(seen == std::vector<std::string>{"small", "large", ""});
        REQUIRE(Body(fixture.Requests().at(2)).at("run_ids") == json{ids[0], ids[2], ids[3]});
    }
}

TEST_CASE("GetBestResults passes an exception from a predicate unchanged", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const auto info = MakeTuningInfo();

    struct PredicateException : ktt::KttException
    {
        PredicateException() :
            KttException("Invalid argument")
        {}
    };

    SECTION("a KttException from the device predicate keeps its type and message")
    {
        fixture.http->EnqueueResponse(200, SpaceFound);
        fixture.http->EnqueueResponse(200, RunPage(RunIds(1)));

        const GetBestResultsQuery query{info.spaceInfo,
            [](const ktt::db::DeviceInfo &) -> bool { throw PredicateException(); }, std::nullopt, 10};
        REQUIRE_THROWS_AS(db.GetBestResults(query), PredicateException);
        REQUIRE(fixture.Requests().size() == 2);
    }

    SECTION("a KttException from the input predicate keeps its message")
    {
        fixture.http->EnqueueResponse(200, SpaceFound);
        fixture.http->EnqueueResponse(200, json{{"runs", {RunJson(RunIds(1)[0], "input")}}}.dump());

        const GetBestResultsQuery query{info.spaceInfo, std::nullopt,
            [](const std::string &) -> bool { throw ktt::KttException("Invalid argument"); }, 10};
        REQUIRE(ExceptionText([&db, &query] { db.GetBestResults(query); }) == "Invalid argument");
    }

    SECTION("another exception type passes too")
    {
        fixture.http->EnqueueResponse(200, SpaceFound);
        fixture.http->EnqueueResponse(200, RunPage(RunIds(1)));

        const GetBestResultsQuery query{info.spaceInfo,
            [](const ktt::db::DeviceInfo &) -> bool { throw std::runtime_error("user error"); }, std::nullopt, 10};
        REQUIRE_THROWS_WITH(db.GetBestResults(query), "user error");
    }
}

TEST_CASE("GetBestResults merges the pages by kernel duration and trims to the limit", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const auto info = MakeTuningInfo();
    const auto ids = RunIds(501);

    fixture.http->EnqueueResponse(200, SpaceFound);
    fixture.http->EnqueueResponse(200, RunPage({ids.begin(), ids.begin() + 500}));

    // Page 1 has a 2 ms result named "first", page 2 one named "second": the tie keeps the earlier page first.
    json first = ResultList({1 * Millisecond, 4 * Millisecond});
    first["results"].insert(first["results"].begin() + 1,
        ResultOut(MakeResult("first", 1, 2 * Millisecond), ktt::OutputFormat::JSON, ktt::TimeUnit::Milliseconds));
    fixture.http->EnqueueResponse(200, first.dump());
    fixture.http->EnqueueResponse(200, RunPage({ids[500]}));

    json second{{"results",
        {ResultOut(MakeResult("second", 2, 2 * Millisecond), ktt::OutputFormat::JSON, ktt::TimeUnit::Seconds),
            ResultOut(MakeResult("kernel", 3, 3 * Millisecond), ktt::OutputFormat::JSON_T4, ktt::TimeUnit::Microseconds)}}};
    fixture.http->EnqueueResponse(200, second.dump());

    const auto results = db.GetBestResults(GetBestResultsQuery{info.spaceInfo, std::nullopt, std::nullopt, 4});

    REQUIRE(Durations(results)
        == std::vector<ktt::Nanoseconds>{1 * Millisecond, 2 * Millisecond, 2 * Millisecond, 3 * Millisecond});
    REQUIRE(results[1].GetKernelName() == "first");
    REQUIRE(results[2].GetKernelName() == "second");
}

// ---------------------------------------------------------------------------------------------------------------------
// GetStatsForSource (R4.4)

TEST_CASE("GetStatsForSource maps the counts and 404", "[Database][Remote]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();

    fixture.http->EnqueueResponse(200, R"({"space_count":3,"device_count":2,"run_count":41,"result_count":20992})");
    const auto stats = db.GetStatsForSource(14695981039346656037ULL);

    REQUIRE(stats.has_value());
    REQUIRE(stats->spaceCount == 3);
    REQUIRE(stats->deviceCount == 2);
    REQUIRE(stats->runCount == 41);
    REQUIRE(stats->resultCount == 20992);
    REQUIRE_FALSE(stats->name.has_value());

    const auto requests = fixture.Requests();
    REQUIRE(requests[0].method == "GET");
    REQUIRE(requests[0].url == AppUrl + "/source/14695981039346656037/stats");
    RequireCommonHeaders(requests[0], false);

    fixture.http->EnqueueResponse(404, R"({"error":"source not found"})");
    REQUIRE_FALSE(db.GetStatsForSource(222).has_value());

    fixture.http->EnqueueResponse(200, R"({"space_count":-1,"device_count":2,"run_count":41,"result_count":0})");
    REQUIRE_THROWS_AS(db.GetStatsForSource(222), ktt::KttException);

    fixture.http->EnqueueResponse(200, R"({"space_count":1})");
    REQUIRE_THROWS_AS(db.GetStatsForSource(222), ktt::KttException);
}

#endif // KTT_REMOTE_DATABASE
