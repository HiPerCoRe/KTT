#include <catch.hpp>

#if defined(KTT_REMOTE_DATABASE)

// R7 / R8.1: RemoteDatabase::QueryResults (POST /result/query) against a scripted fake HttpClient.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
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
using ktt::db::RemoteDatabase;
using ktt::db::RemoteDatabaseOptions;
using ktt::db::ResultQuery;
using ktt::db::ResultQueryItem;
using ktt::db::UuidGenerator;
using ktt::db::test::FakeHttpClient;
using ktt::db::test::MakeResult;
using ktt::db::test::Millisecond;
using ktt::db::test::RecordedRequest;

namespace
{

const std::string ApiKey = "kttak-query-test-key";
const std::string BaseUrl = RemoteDatabase::GetHost();
const std::string QueryUrl = BaseUrl + "/ktt-db/app/result/query";

struct QueryFixture
{
    std::shared_ptr<FakeHttpClient> http = std::make_shared<FakeHttpClient>();

    // No retries unless a test asks for them: a retry sleeps for real (250 ms).
    RemoteDatabase MakeDatabase(const uint32_t maxRetries = 0) const
    {
        RemoteDatabaseOptions options;
        options.maxRetries = maxRetries;
        return RemoteDatabase(ApiKey, options, http);
    }
};

std::string QueryExceptionText(const std::function<void()> &function)
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

json CpuDevice()
{
    return json{{"id", 3}, {"name", "Test CPU"}, {"vendor", "Test Vendor"}, {"type", "CPU"}, {"compute_api", "cpp"},
        {"device_identifier", nullptr}, {"driver_version", nullptr}, {"extensions", nullptr},
        {"cuda_compute_capability_major", nullptr}, {"cuda_compute_capability_minor", nullptr}};
}

json CudaDevice()
{
    return json{{"id", 5}, {"name", "NVIDIA GeForce RTX 3090"}, {"vendor", "NVIDIA"}, {"type", "GPU"},
        {"compute_api", "cuda"}, {"device_identifier", "GPU-6c3f1a2b"}, {"driver_version", "550.54.14"},
        {"extensions", nullptr}, {"cuda_compute_capability_major", 8}, {"cuda_compute_capability_minor", 6}};
}

// One item of a ResultQueryResponse, as the application API sends it with $select=run,device,source[,result]: the
// payload serialized in the given format and in the time unit of its run.
json QueryItem(const ktt::KernelResult &result, const ktt::OutputFormat format, const ktt::TimeUnit unit,
    const std::string &runId, const json &inputData, const json &device, const std::string &sourceFingerprint,
    const bool withResult = true)
{
    json configuration = json::object();

    for (const auto &pair : result.GetConfiguration().GetPairs())
        configuration[pair.GetName()] = std::get<uint64_t>(pair.GetValue());

    json item{{"id", 99121}, {"duration", result.GetKernelDuration()}, {"total_duration", result.GetTotalDuration()},
        {"configuration", configuration},
        {"run",
            {{"id", runId}, {"input_data", inputData},
                {"output_format", format == ktt::OutputFormat::JSON_T4 ? "json_t4" : "json"},
                {"time_unit", ktt::db::TimeUnitToName(unit)}, {"created_at", "2026-11-06T10:12:00Z"}}},
        {"device", device}, {"source", {{"id", 14}, {"fingerprint", sourceFingerprint}}}};

    if (withResult)
    {
        const ktt::db::TimeUnitGuard guard(unit);
        item["result"] = json::parse(ktt::db::ResultSerialization::SerializeResult(result, format, 0));
    }

    return item;
}

json DefaultItem(const bool withResult = true)
{
    return QueryItem(MakeResult("kernel", 64, 3 * Millisecond), ktt::OutputFormat::JSON, ktt::TimeUnit::Milliseconds,
        UuidGenerator::ToString(UuidGenerator::GenerateUuidV7()), nullptr, CpuDevice(), "222", withResult);
}

std::string Page(const std::vector<json> &items)
{
    return json{{"items", items}}.dump();
}

void RequireQueryRequest(const RecordedRequest &request)
{
    REQUIRE(request.method == "POST");
    REQUIRE(request.url == QueryUrl);
    REQUIRE(request.Header("x-api-key") == ApiKey);
    REQUIRE(UuidGenerator::GetVersion(UuidGenerator::Parse(request.Header("correlation-id").value())) == 4);
    REQUIRE(request.Header("content-type") == "application/json");
}

} // namespace

TEST_CASE("QueryResults posts the options and reads the items with their payloads", "[Database][Remote]")
{
    QueryFixture fixture;
    const auto db = fixture.MakeDatabase();

    const std::string runA = UuidGenerator::ToString(UuidGenerator::GenerateUuidV7());
    std::string runAUpper = runA;
    std::transform(runAUpper.begin(), runAUpper.end(), runAUpper.begin(),
        [](const unsigned char c) { return static_cast<char>(std::toupper(c)); });
    const std::string runB = UuidGenerator::ToString(UuidGenerator::GenerateUuidV7());

    json first = QueryItem(MakeResult("a", 64, 2 * Millisecond), ktt::OutputFormat::JSON, ktt::TimeUnit::Microseconds,
        runAUpper, "atoms=4096", CudaDevice(), "14695981039346656037");
    // Every JSON type a configuration value can have.
    first["configuration"] = json{{"block_size", 64}, {"offset", -3}, {"ratio", 0.5}, {"shared", true},
        {"variant", "abc"}};
    const json second = QueryItem(MakeResult("b", 32, 5 * Millisecond), ktt::OutputFormat::JSON_T4,
        ktt::TimeUnit::Seconds, runB, nullptr, CpuDevice(), "222");
    fixture.http->EnqueueResponse(200, json{{"items", {first, second}}}.dump());

    // A unit other than the payloads' units is active: each payload is read in the unit of its run.
    const ktt::db::TimeUnitGuard active(ktt::TimeUnit::Nanoseconds);

    ResultQuery query;
    query.filter = "run/input_data eq 'it''s' and device/name eq 'Grafick\xC3\xA1 karta' and duration lt 100000";
    query.orderBy = "configuration/block_size desc, duration asc";
    query.top = 10;
    query.skip = 20;
    const auto items = db.QueryResults(query);

    const auto requests = fixture.http->Requests();
    REQUIRE(requests.size() == 1);
    RequireQueryRequest(requests[0]);
    // The body matches ResultQueryRequest exactly: no "$", no count or distinct, and the filter unchanged.
    REQUIRE(json::parse(requests[0].body) == json{{"filter", query.filter},
        {"orderby", "configuration/block_size desc, duration asc"}, {"top", 10}, {"skip", 20},
        {"select", "run,device,source,result"}});

    REQUIRE(items.size() == 2);

    const ResultQueryItem &a = items[0];
    REQUIRE(a.duration == 2 * Millisecond);
    REQUIRE(a.totalDuration == 2 * Millisecond);
    REQUIRE(a.configuration.size() == 5);
    REQUIRE(std::get<uint64_t>(a.configuration.at("block_size")) == 64);
    REQUIRE(std::get<int64_t>(a.configuration.at("offset")) == -3);
    REQUIRE(std::get<double>(a.configuration.at("ratio")) == 0.5);
    REQUIRE(std::get<bool>(a.configuration.at("shared")));
    REQUIRE(std::get<std::string>(a.configuration.at("variant")) == "abc");
    REQUIRE(a.runId == runA);
    REQUIRE(a.inputData == "atoms=4096");
    REQUIRE(a.device.name == "NVIDIA GeForce RTX 3090");
    REQUIRE(a.device.vendor == "NVIDIA");
    REQUIRE(a.device.type == "GPU");
    REQUIRE(a.device.computeApi == ktt::ComputeApi::CUDA);
    REQUIRE(a.device.deviceIdentifier == "GPU-6c3f1a2b");
    REQUIRE(a.device.driverVersion == "550.54.14");
    REQUIRE_FALSE(a.device.extensions.has_value());
    REQUIRE(a.device.cudaComputeCapabilityMajor == 8u);
    REQUIRE(a.device.cudaComputeCapabilityMinor == 6u);
    REQUIRE(a.sourceFingerprint == "14695981039346656037");
    REQUIRE(a.result.has_value());
    REQUIRE(a.result->GetKernelName() == "a");
    REQUIRE(a.result->GetKernelDuration() == 2 * Millisecond);
    REQUIRE(a.result->GetConfiguration().GetPairs().at(0).GetName() == "block_size");

    const ResultQueryItem &b = items[1];
    REQUIRE(b.duration == 5 * Millisecond);
    REQUIRE(b.configuration.size() == 1);
    REQUIRE(std::get<uint64_t>(b.configuration.at("block_size")) == 32);
    REQUIRE(b.runId == runB);
    REQUIRE_FALSE(b.inputData.has_value());
    REQUIRE(b.device.computeApi == ktt::ComputeApi::Cpp);
    REQUIRE(b.device.driverVersion.empty());
    REQUIRE_FALSE(b.device.deviceIdentifier.has_value());
    REQUIRE_FALSE(b.device.cudaComputeCapabilityMajor.has_value());
    REQUIRE(b.sourceFingerprint == "222");
    REQUIRE(b.result.has_value());
    // JSON T4 keeps no kernel name, so the configuration identifies the payload.
    REQUIRE(std::get<uint64_t>(b.result->GetConfiguration().GetPairs().at(0).GetValue()) == 32);
    REQUIRE(b.result->GetKernelDuration() == 5 * Millisecond);

    // The active unit is restored after reading the payloads.
    REQUIRE(ktt::TimeConfiguration::GetInstance().GetTimeUnit() == ktt::TimeUnit::Nanoseconds);
}

TEST_CASE("QueryResults without the payload selects no result group", "[Database][Remote]")
{
    QueryFixture fixture;
    const auto db = fixture.MakeDatabase();

    SECTION("the defaults")
    {
        fixture.http->EnqueueResponse(200, Page({}));
        REQUIRE(db.QueryResults({}).empty());
        REQUIRE(json::parse(fixture.http->Requests().at(0).body) == json{{"filter", ""}, {"orderby", "duration asc"},
            {"top", 50}, {"skip", 0}, {"select", "run,device,source,result"}});
    }

    SECTION("includeResult false")
    {
        fixture.http->EnqueueResponse(200, Page({DefaultItem(false)}));
        ResultQuery query;
        query.filter = "device/compute_api eq 'cpp'";
        query.includeResult = false;
        const auto items = db.QueryResults(query);

        REQUIRE(json::parse(fixture.http->Requests().at(0).body).at("select") == "run,device,source");
        REQUIRE(items.size() == 1);
        REQUIRE_FALSE(items[0].result.has_value());
        // The promoted fields are filled without the payload.
        REQUIRE(items[0].duration == 3 * Millisecond);
        REQUIRE(items[0].totalDuration == 3 * Millisecond);
        REQUIRE(std::get<uint64_t>(items[0].configuration.at("block_size")) == 64);
        REQUIRE(items[0].device.name == "Test CPU");
        REQUIRE(items[0].sourceFingerprint == "222");
    }
}

TEST_CASE("QueryResults checks top and skip on the client", "[Database][Remote]")
{
    QueryFixture fixture;
    const auto db = fixture.MakeDatabase();
    ResultQuery query;

    SECTION("top 0 returns nothing without a request")
    {
        query.top = 0;
        REQUIRE(db.QueryResults(query).empty());
        REQUIRE(fixture.http->Requests().empty());
    }

    SECTION("top above 1000 throws without a request")
    {
        query.top = 1001;
        REQUIRE(QueryExceptionText([&] { db.QueryResults(query); }).find("top must be at most 1000")
            != std::string::npos);
        REQUIRE(fixture.http->Requests().empty());
    }

    SECTION("skip above 100000 throws without a request")
    {
        query.skip = 100001;
        REQUIRE(QueryExceptionText([&] { db.QueryResults(query); }).find("skip at most 100000")
            != std::string::npos);
        REQUIRE(fixture.http->Requests().empty());
    }

    SECTION("the maximums are sent")
    {
        fixture.http->EnqueueResponse(200, Page({}));
        query.top = 1000;
        query.skip = 100000;
        REQUIRE(db.QueryResults(query).empty());

        const json body = json::parse(fixture.http->Requests().at(0).body);
        REQUIRE(body.at("top") == 1000);
        REQUIRE(body.at("skip") == 100000);
    }
}

TEST_CASE("QueryResults maps error responses to exceptions", "[Database][Remote]")
{
    QueryFixture fixture;
    const auto db = fixture.MakeDatabase(3);
    const auto query = [&db]
    {
        ResultQuery q;
        q.filter = "device/colour eq 'red'";
        db.QueryResults(q);
    };

    SECTION("a 400 has the server's message and is not retried")
    {
        fixture.http->EnqueueResponse(400, R"({"error":"$filter: unknown field 'device/colour' at position 0"})");
        const std::string text = QueryExceptionText(query);
        REQUIRE(text.find("RemoteDatabase::QueryResults") != std::string::npos);
        REQUIRE(text.find("HTTP 400") != std::string::npos);
        REQUIRE(text.find("$filter: unknown field 'device/colour' at position 0") != std::string::npos);
        REQUIRE(fixture.http->Requests().size() == 1);
    }

    SECTION("a 503 query timeout is not retried")
    {
        fixture.http->EnqueueResponse(503, R"({"error":"query timed out; narrow the filter","code":"query_timeout"})");
        const std::string text = QueryExceptionText(query);
        REQUIRE(text.find("HTTP 503") != std::string::npos);
        REQUIRE(text.find("query timed out; narrow the filter") != std::string::npos);
        REQUIRE(text.find("query_timeout") != std::string::npos);
        REQUIRE(fixture.http->Requests().size() == 1);
    }

    SECTION("a 404 is an error, not an empty result")
    {
        fixture.http->EnqueueResponse(404, R"({"error":"not found"})");
        REQUIRE(QueryExceptionText(query).find("HTTP 404") != std::string::npos);
        REQUIRE(fixture.http->Requests().size() == 1);
    }

    SECTION("403 and 500 throw")
    {
        fixture.http->EnqueueResponse(403, R"({"error":"no usable realm"})");
        fixture.http->EnqueueResponse(500, "");
        REQUIRE(QueryExceptionText(query).find("no usable realm") != std::string::npos);
        REQUIRE(QueryExceptionText(query).find("HTTP 500") != std::string::npos);
    }

    SECTION("the API key is not in the message")
    {
        fixture.http->EnqueueResponse(401, R"({"error":"invalid api key )" + ApiKey + R"("})");
        const std::string text = QueryExceptionText(query);
        REQUIRE(text.find("HTTP 401") != std::string::npos);
        REQUIRE(text.find(ApiKey) == std::string::npos);
        REQUIRE(text.find("<api key>") != std::string::npos);
    }

    SECTION("a network error after the retries")
    {
        fixture.http->EnqueueTransportError("Could not resolve host", false);
        REQUIRE(QueryExceptionText(query).find("Could not resolve host") != std::string::npos);
        REQUIRE(fixture.http->Requests().size() == 1);
    }
}

TEST_CASE("QueryResults retries a busy server with the same request", "[Database][Remote]")
{
    QueryFixture fixture;
    const auto db = fixture.MakeDatabase(1);
    fixture.http->EnqueueResponse(503, R"({"error":"the server is busy; try again later"})");
    fixture.http->EnqueueResponse(200, Page({DefaultItem()}));

    const auto items = db.QueryResults({});

    REQUIRE(items.size() == 1);
    const auto requests = fixture.http->Requests();
    REQUIRE(requests.size() == 2);
    REQUIRE(requests[0].body == requests[1].body);
    REQUIRE(requests[0].Header("correlation-id") == requests[1].Header("correlation-id"));
}

TEST_CASE("QueryResults rejects malformed responses", "[Database][Remote]")
{
    const std::vector<std::pair<std::string, std::function<void(json &)>>> cases{
        {"no items", [](json &page) { page.erase("items"); }},
        {"items not an array", [](json &page) { page["items"] = json::object(); }},
        {"a negative duration", [](json &page) { page["items"][0]["duration"] = -1; }},
        {"a fractional duration", [](json &page) { page["items"][0]["total_duration"] = 1.5; }},
        {"a configuration that is not an object", [](json &page) { page["items"][0]["configuration"] = json::array(); }},
        {"a configuration value that is an array",
            [](json &page) { page["items"][0]["configuration"]["block_size"] = json::array({1}); }},
        {"a configuration value that is null",
            [](json &page) { page["items"][0]["configuration"]["block_size"] = nullptr; }},
        {"no run group", [](json &page) { page["items"][0].erase("run"); }},
        {"an invalid run id", [](json &page) { page["items"][0]["run"]["id"] = "not-a-uuid"; }},
        {"no device group", [](json &page) { page["items"][0].erase("device"); }},
        {"an unknown compute API", [](json &page) { page["items"][0]["device"]["compute_api"] = "metal"; }},
        {"an invalid fingerprint", [](json &page) { page["items"][0]["source"]["fingerprint"] = "-1"; }},
        {"a numeric fingerprint", [](json &page) { page["items"][0]["source"]["fingerprint"] = 222; }},
        {"no payload", [](json &page) { page["items"][0].erase("result"); }},
        {"a payload that is not an object", [](json &page) { page["items"][0]["result"] = "{}"; }},
        {"an unknown output format", [](json &page) { page["items"][0]["run"]["output_format"] = "xml"; }},
        {"an unknown time unit", [](json &page) { page["items"][0]["run"]["time_unit"] = "minutes"; }}};

    for (const auto &[name, mutate] : cases)
    {
        CAPTURE(name);
        QueryFixture fixture;
        const auto db = fixture.MakeDatabase();

        json page{{"items", {DefaultItem()}}};
        mutate(page);
        fixture.http->EnqueueResponse(200, page.dump());

        const std::string text = QueryExceptionText([&db] { db.QueryResults({}); });
        CAPTURE(text);
        REQUIRE(text.rfind("RemoteDatabase::QueryResults failed", 0) == 0);
    }

    QueryFixture fixture;
    const auto db = fixture.MakeDatabase();
    fixture.http->EnqueueResponse(200, "<html>proxy</html>");
    REQUIRE(QueryExceptionText([&db] { db.QueryResults({}); }).find("not a JSON object") != std::string::npos);
}

#endif // KTT_REMOTE_DATABASE
