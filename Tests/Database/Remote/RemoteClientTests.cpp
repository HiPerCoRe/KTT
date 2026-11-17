#include <catch.hpp>

#if defined(KTT_REMOTE_DATABASE)

#include <chrono>
#include <functional>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <Api/KttException.h>
#include <Database/Remote/Utility/RemoteClient.h>

#include "FakeHttpClient.h"

using ktt::db::HttpResponse;
using ktt::db::RemoteClient;
using ktt::db::RetryPolicy;
using ktt::db::test::FakeHttpClient;

namespace
{

const std::string ApiKey = "kttak-secret-test-key";
const std::string BaseUrl = "https://ktt.example.org";

struct ClientFixture
{
    std::shared_ptr<FakeHttpClient> http = std::make_shared<FakeHttpClient>();
    std::vector<std::chrono::milliseconds> sleeps;

    RemoteClient MakeClient(const uint32_t maxRetries = 3, const std::string &baseUrl = BaseUrl)
    {
        return RemoteClient(baseUrl, ApiKey, maxRetries, http,
            [this](const std::chrono::milliseconds duration) { sleeps.push_back(duration); });
    }
};

std::string ExceptionText(const std::function<void()> &action)
{
    try
    {
        action();
    }
    catch (const ktt::KttException &exception)
    {
        return exception.what();
    }

    FAIL("expected a KttException");
    return {};
}

bool IsUuidV4(const std::string &text)
{
    static const std::regex pattern("^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$");
    return std::regex_match(text, pattern);
}

} // namespace

TEST_CASE("RemoteClient sends to /ktt-db/app with the API key, a correlation id and the content type",
    "[Database][Remote]")
{
    ClientFixture fixture;
    const RemoteClient client = fixture.MakeClient();

    SECTION("a request with a body")
    {
        const HttpResponse response = client.Send("POST", "/run", R"({"a":1})", RetryPolicy::Idempotent);
        REQUIRE(response.status == 200);

        const auto requests = fixture.http->Requests();
        REQUIRE(requests.size() == 1);
        REQUIRE(requests[0].method == "POST");
        REQUIRE(requests[0].url == "https://ktt.example.org/ktt-db/app/run");
        REQUIRE(requests[0].body == R"({"a":1})");
        REQUIRE(requests[0].Header("x-api-key") == ApiKey);
        REQUIRE(requests[0].Header("content-type") == "application/json");
        REQUIRE(requests[0].Header("accept") == "application/json");
        REQUIRE(IsUuidV4(requests[0].Header("correlation-id").value_or("")));
    }

    SECTION("a request without a body has no content type")
    {
        client.Send("GET", "/source/123/stats", "", RetryPolicy::Idempotent);

        const auto requests = fixture.http->Requests();
        REQUIRE(requests.size() == 1);
        REQUIRE(requests[0].method == "GET");
        REQUIRE(requests[0].url == "https://ktt.example.org/ktt-db/app/source/123/stats");
        REQUIRE(requests[0].body.empty());
        REQUIRE_FALSE(requests[0].Header("content-type").has_value());
        REQUIRE(requests[0].Header("x-api-key") == ApiKey);
    }

    SECTION("every request has a new correlation id")
    {
        client.Send("GET", "/a", "", RetryPolicy::Idempotent);
        client.Send("GET", "/b", "", RetryPolicy::Idempotent);

        const auto requests = fixture.http->Requests();
        REQUIRE(requests.size() == 2);
        REQUIRE(requests[0].Header("correlation-id") != requests[1].Header("correlation-id"));
    }
}

TEST_CASE("RemoteClient removes trailing slashes from the base URL", "[Database][Remote]")
{
    ClientFixture fixture;
    const RemoteClient client = fixture.MakeClient(3, "http://localhost:8080//");

    REQUIRE(client.GetBaseUrl() == "http://localhost:8080");
    client.Send("GET", "/x", "", RetryPolicy::Idempotent);
    REQUIRE(fixture.http->Requests()[0].url == "http://localhost:8080/ktt-db/app/x");
}

TEST_CASE("RemoteClient rejects an invalid base URL or API key without showing the key", "[Database][Remote]")
{
    const auto http = std::make_shared<FakeHttpClient>();

    for (const std::string url : {"", "ktt.example.org", "ftp://ktt.example.org", "https://", "https:///",
             "https://ktt.example.org/a b", "https://ktt.example.org/?x=1", "https://ktt.example.org/#x"})
    {
        CAPTURE(url);
        REQUIRE_THROWS_AS(RemoteClient(url, ApiKey, 3, http), ktt::KttException);
    }

    REQUIRE_NOTHROW(RemoteClient("HTTPS://ktt.example.org/base", ApiKey, 3, http));

    REQUIRE_THROWS_AS(RemoteClient(BaseUrl, "", 3, http), ktt::KttException);

    const std::string badKey = ApiKey + "\r\ninjected: header";
    const std::string text = ExceptionText([&] { static_cast<void>(RemoteClient(BaseUrl, badKey, 3, http)); });
    REQUIRE(text.find(ApiKey) == std::string::npos);

    REQUIRE_THROWS_AS(RemoteClient(BaseUrl, ApiKey, 3, nullptr), ktt::KttException);
    REQUIRE(http->Requests().empty());
}

TEST_CASE("RemoteClient leaves the rejected base URL out of the exception message", "[Database][Remote]")
{
    const auto http = std::make_shared<FakeHttpClient>();

    SECTION("the API key passed as the URL (swapped arguments)")
    {
        const std::string text = ExceptionText([&] { static_cast<void>(RemoteClient(ApiKey, BaseUrl, 3, http)); });
        REQUIRE(text.find(ApiKey) == std::string::npos);
        REQUIRE(text.find("secret") == std::string::npos);
    }

    SECTION("a URL with credentials or a token in its query")
    {
        for (const std::string &url : {"https://user:" + ApiKey + "@ktt.example.org/?x=1",
                 "https://ktt.example.org/?key=" + ApiKey, "https://ktt.example.org/" + ApiKey + " x"})
        {
            CAPTURE(url);
            const std::string text = ExceptionText([&] { static_cast<void>(RemoteClient(url, ApiKey, 3, http)); });
            REQUIRE(text.find(ApiKey) == std::string::npos);
            REQUIRE(text.find("secret") == std::string::npos);
        }
    }

    REQUIRE(http->Requests().empty());
}

TEST_CASE("RemoteClient retries connection errors, timeouts and 502 / 503 / 504 with exponential backoff",
    "[Database][Remote]")
{
    ClientFixture fixture;
    const RemoteClient client = fixture.MakeClient(3);
    const std::string body = R"({"run_id":"0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b17"})";

    fixture.http->EnqueueTransportError("Timeout was reached", true);
    fixture.http->EnqueueResponse(502);
    fixture.http->EnqueueResponse(503, R"({"error":"the server is busy; try again later"})");
    fixture.http->EnqueueResponse(201, R"({"run_id":"x"})");

    const HttpResponse response = client.Send("POST", "/run", body, RetryPolicy::Idempotent);

    REQUIRE(response.status == 201);
    REQUIRE(response.body == R"({"run_id":"x"})");

    const auto requests = fixture.http->Requests();
    REQUIRE(requests.size() == 4);

    std::set<std::string> correlationIds;

    for (const auto &request : requests)
    {
        // The same body (with the same run id) is resent on every attempt.
        REQUIRE(request.body == body);
        REQUIRE(request.Header("x-api-key") == ApiKey);
        correlationIds.insert(request.Header("correlation-id").value_or(""));
    }

    // One logical request: every attempt carries the same correlation id.
    REQUIRE(correlationIds.size() == 1);
    REQUIRE(IsUuidV4(*correlationIds.begin()));
    REQUIRE(fixture.sleeps == std::vector<std::chrono::milliseconds>{
        std::chrono::milliseconds(250), std::chrono::milliseconds(500), std::chrono::milliseconds(1000)});
}

TEST_CASE("RemoteClient retries 504", "[Database][Remote]")
{
    ClientFixture fixture;
    const RemoteClient client = fixture.MakeClient(3);
    fixture.http->EnqueueResponse(504);

    REQUIRE(client.Send("GET", "/x", "", RetryPolicy::Idempotent).status == 200);
    REQUIRE(fixture.http->Requests().size() == 2);
}

TEST_CASE("RemoteClient does not retry other statuses", "[Database][Remote]")
{
    for (const long status : {400L, 401L, 403L, 404L, 409L, 413L, 500L})
    {
        CAPTURE(status);
        ClientFixture fixture;
        const RemoteClient client = fixture.MakeClient(3);
        fixture.http->EnqueueResponse(status, R"({"error":"nope"})");

        const HttpResponse response = client.Send("POST", "/run", "{}", RetryPolicy::Idempotent);

        REQUIRE(response.status == status);
        REQUIRE(response.body == R"({"error":"nope"})");
        REQUIRE(fixture.http->Requests().size() == 1);
        REQUIRE(fixture.sleeps.empty());
    }
}

TEST_CASE("RemoteClient does not retry a 503 query timeout", "[Database][Remote]")
{
    ClientFixture fixture;
    const RemoteClient client = fixture.MakeClient(3);
    fixture.http->EnqueueResponse(503, R"({"error":"query timed out; narrow the filter","code":"query_timeout"})");

    REQUIRE(client.Send("POST", "/result/query", "{}", RetryPolicy::Idempotent).status == 503);
    REQUIRE(fixture.http->Requests().size() == 1);
}

TEST_CASE("RemoteClient returns the last 5xx response when the retries run out", "[Database][Remote]")
{
    ClientFixture fixture;
    const RemoteClient client = fixture.MakeClient(2);

    for (int i = 0; i < 5; ++i)
        fixture.http->EnqueueResponse(503, R"({"error":"busy"})");

    const HttpResponse response = client.Send("GET", "/x", "", RetryPolicy::Idempotent);

    REQUIRE(response.status == 503);
    REQUIRE(fixture.http->Requests().size() == 3);
    REQUIRE(fixture.sleeps.size() == 2);
}

TEST_CASE("RemoteClient throws KttException without the API key when no response arrives", "[Database][Remote]")
{
    SECTION("retryable errors until the retries run out")
    {
        ClientFixture fixture;
        const RemoteClient client = fixture.MakeClient(1);
        fixture.http->EnqueueTransportError("Couldn't connect to server", true);
        fixture.http->EnqueueTransportError("Couldn't connect to server", true);

        const std::string text = ExceptionText([&] { client.Send("POST", "/run", "{}", RetryPolicy::Idempotent); });

        REQUIRE(fixture.http->Requests().size() == 2);
        REQUIRE(text.find("POST /run") != std::string::npos);
        REQUIRE(text.find("Couldn't connect to server") != std::string::npos);
        REQUIRE(text.find("2 attempts") != std::string::npos);
        REQUIRE(text.find(ApiKey) == std::string::npos);
    }

    SECTION("a non-retryable error is not retried")
    {
        ClientFixture fixture;
        const RemoteClient client = fixture.MakeClient(3);
        fixture.http->EnqueueTransportError("SSL peer certificate was not OK", false);

        const std::string text = ExceptionText([&] { client.Send("GET", "/x", "", RetryPolicy::Idempotent); });

        REQUIRE(fixture.http->Requests().size() == 1);
        REQUIRE(fixture.sleeps.empty());
        REQUIRE(text.find("SSL peer certificate was not OK") != std::string::npos);
        REQUIRE(text.find(ApiKey) == std::string::npos);
    }
}

TEST_CASE("RemoteClient sends a RetryPolicy::Never request once", "[Database][Remote]")
{
    ClientFixture fixture;
    const RemoteClient client = fixture.MakeClient(3);

    fixture.http->EnqueueResponse(503);
    REQUIRE(client.Send("POST", "/x", "{}", RetryPolicy::Never).status == 503);

    fixture.http->EnqueueTransportError("Timeout was reached", true);
    REQUIRE_THROWS_AS(client.Send("POST", "/x", "{}", RetryPolicy::Never), ktt::KttException);

    REQUIRE(fixture.http->Requests().size() == 2);
    REQUIRE(fixture.sleeps.empty());
}

TEST_CASE("RemoteClient with maxRetries 0 sends every request once", "[Database][Remote]")
{
    ClientFixture fixture;
    const RemoteClient client = fixture.MakeClient(0);
    fixture.http->EnqueueResponse(502);

    REQUIRE(client.Send("GET", "/x", "", RetryPolicy::Idempotent).status == 502);
    REQUIRE(fixture.http->Requests().size() == 1);
}

TEST_CASE("RemoteClient backoff doubles up to a cap", "[Database][Remote]")
{
    REQUIRE(RemoteClient::GetBackoff(1) == std::chrono::milliseconds(250));
    REQUIRE(RemoteClient::GetBackoff(2) == std::chrono::milliseconds(500));
    REQUIRE(RemoteClient::GetBackoff(3) == std::chrono::milliseconds(1000));
    REQUIRE(RemoteClient::GetBackoff(6) == std::chrono::milliseconds(8000));
    REQUIRE(RemoteClient::GetBackoff(1000) == std::chrono::milliseconds(8000));
}

TEST_CASE("RemoteClient can be used from several threads at once", "[Database][Remote]")
{
    const auto http = std::make_shared<FakeHttpClient>();
    const RemoteClient client(BaseUrl, ApiKey, 3, http, [](std::chrono::milliseconds) {});

    std::vector<std::thread> threads;

    for (int t = 0; t < 8; ++t)
    {
        threads.emplace_back([&client] {
            for (int i = 0; i < 50; ++i)
                client.Send("GET", "/x", "", RetryPolicy::Idempotent);
        });
    }

    for (auto &thread : threads)
        thread.join();

    const auto requests = http->Requests();
    REQUIRE(requests.size() == 400);

    std::set<std::string> correlationIds;

    for (const auto &request : requests)
        correlationIds.insert(request.Header("correlation-id").value_or(""));

    REQUIRE(correlationIds.size() == 400);
}

#endif // KTT_REMOTE_DATABASE
