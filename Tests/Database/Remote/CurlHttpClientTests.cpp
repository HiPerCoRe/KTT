#include <catch.hpp>

// These tests drive the real libcurl client against a one-shot HTTP server inside the test process, bound to
// 127.0.0.1. Nothing leaves the machine. POSIX sockets only, hence not on MSVC.
#if defined(KTT_REMOTE_DATABASE) && !defined(_MSC_VER)

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <chrono>
#include <memory>
#include <netinet/in.h>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include <Api/KttException.h>
#include <Database/Remote/Http/CurlHttpClient.h>
#include <Database/Remote/Utility/RemoteClient.h>

#include "../Local/DefaultLocation/DefaultLocationTestHelpers.h"

using ktt::db::CurlHttpClient;
using ktt::db::HttpResponse;
using ktt::db::HttpTransportError;
using ktt::db::RemoteDatabaseOptions;
using ktt::db::test::ScopedEnvironmentVariable;

namespace
{

std::string ToLower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

// Opens a listening socket on 127.0.0.1 with a free port.
int ListenOnLoopback(int &port)
{
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(listener >= 0);

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    REQUIRE(bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0);
    REQUIRE(listen(listener, 1) == 0);

    socklen_t length = sizeof(address);
    REQUIRE(getsockname(listener, reinterpret_cast<sockaddr *>(&address), &length) == 0);
    port = ntohs(address.sin_port);
    return listener;
}

// Accepts one connection, reads one request and answers with a canned response (or with nothing, until the client
// gives up). Every wait is bounded, so a failing test cannot hang.
class OneShotServer
{
public:
    explicit OneShotServer(std::string response) :
        m_Response(std::move(response))
    {
        m_Listener = ListenOnLoopback(m_Port);
        m_Thread = std::thread([this] { Serve(); });
    }

    ~OneShotServer()
    {
        if (m_Thread.joinable())
            m_Thread.join();

        close(m_Listener);
    }

    std::string Url() const
    {
        return "http://127.0.0.1:" + std::to_string(m_Port);
    }

    // The raw request received; waits for the server thread to finish.
    const std::string &Request()
    {
        if (m_Thread.joinable())
            m_Thread.join();

        return m_Request;
    }

private:
    std::string m_Response;
    int m_Listener{-1};
    int m_Port{};
    std::string m_Request;
    std::thread m_Thread;

    static bool WaitReadable(const int socket)
    {
        pollfd descriptor{socket, POLLIN, 0};
        return poll(&descriptor, 1, 5000) == 1;
    }

    void Serve()
    {
        if (!WaitReadable(m_Listener))
            return;

        const int connection = accept(m_Listener, nullptr, nullptr);

        if (connection < 0)
            return;

        char buffer[4096];
        size_t expectedSize = std::string::npos;

        while (m_Request.size() < expectedSize && WaitReadable(connection))
        {
            const ssize_t received = recv(connection, buffer, sizeof(buffer), 0);

            if (received <= 0)
                break;

            m_Request.append(buffer, static_cast<size_t>(received));
            const size_t headerEnd = m_Request.find("\r\n\r\n");

            if (expectedSize == std::string::npos && headerEnd != std::string::npos)
            {
                const std::string headers = ToLower(m_Request.substr(0, headerEnd));
                const size_t lengthPosition = headers.find("content-length:");
                const size_t bodySize = lengthPosition == std::string::npos
                    ? 0 : std::stoul(headers.substr(lengthPosition + 15));
                expectedSize = headerEnd + 4 + bodySize;
            }
        }

        if (!m_Response.empty())
        {
            send(connection, m_Response.data(), m_Response.size(), 0);
        }
        else
        {
            // No answer: wait until the client closes the connection (its timeout).
            while (WaitReadable(connection) && recv(connection, buffer, sizeof(buffer), 0) > 0)
            {}
        }

        close(connection);
    }
};

std::string MakeResponse(const std::string &statusLine, const std::string &body, const std::string &extraHeaders = "")
{
    return "HTTP/1.1 " + statusLine + "\r\nContent-Type: application/json\r\nContent-Length: "
        + std::to_string(body.size()) + "\r\nConnection: close\r\n" + extraHeaders + "\r\n" + body;
}

// A proxy from the environment would take the loopback requests elsewhere.
struct NoProxy
{
    ScopedEnvironmentVariable lower{"no_proxy", std::string("*")};
    ScopedEnvironmentVariable upper{"NO_PROXY", std::string("*")};
};

} // namespace

TEST_CASE("CurlHttpClient sends the method, headers and body through RemoteClient", "[Database][Remote][Curl]")
{
    const NoProxy noProxy;
    OneShotServer server(MakeResponse("201 Created", R"({"source_has_code":false})"));

    const auto http = std::make_shared<CurlHttpClient>(RemoteDatabaseOptions{});
    const ktt::db::RemoteClient client(server.Url(), "kttak-loopback", 0, http);
    const std::string body = R"({"run_id":"0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b17"})";

    const HttpResponse response = client.Send("POST", "/run", body, ktt::db::RetryPolicy::Idempotent);

    REQUIRE(response.status == 201);
    REQUIRE(response.body == R"({"source_has_code":false})");

    const std::string request = server.Request();
    const std::string lower = ToLower(request);
    REQUIRE(request.rfind("POST /ktt-db/app/run HTTP/1.1\r\n", 0) == 0);
    REQUIRE(lower.find("\r\nx-api-key: kttak-loopback\r\n") != std::string::npos);
    REQUIRE(lower.find("\r\ncorrelation-id: ") != std::string::npos);
    REQUIRE(lower.find("\r\ncontent-type: application/json\r\n") != std::string::npos);
    REQUIRE(lower.find("\r\ncontent-length: " + std::to_string(body.size()) + "\r\n") != std::string::npos);
    REQUIRE(lower.find("\r\nexpect:") == std::string::npos);
    REQUIRE(request.size() >= body.size());
    REQUIRE(request.substr(request.size() - body.size()) == body);
}

TEST_CASE("CurlHttpClient sends GET without a body or content type", "[Database][Remote][Curl]")
{
    const NoProxy noProxy;
    OneShotServer server(MakeResponse("200 OK", R"({"spaceCount":1})"));
    CurlHttpClient http(RemoteDatabaseOptions{});

    const HttpResponse response = http.Send("GET", server.Url() + "/ktt-db/app/source/1/stats", {}, "");

    REQUIRE(response.status == 200);
    REQUIRE(response.body == R"({"spaceCount":1})");

    const std::string lower = ToLower(server.Request());
    REQUIRE(lower.rfind("get /ktt-db/app/source/1/stats http/1.1\r\n", 0) == 0);
    REQUIRE(lower.find("content-type") == std::string::npos);
}

TEST_CASE("CurlHttpClient sends PUT with its body", "[Database][Remote][Curl]")
{
    const NoProxy noProxy;
    OneShotServer server(MakeResponse("200 OK", "{}"));
    CurlHttpClient http(RemoteDatabaseOptions{});

    const std::string body = R"({"definitions":[]})";
    const HttpResponse response = http.Send("PUT", server.Url() + "/code", {{"content-type", "application/json"}},
        body);

    REQUIRE(response.status == 200);

    const std::string request = server.Request();
    REQUIRE(request.rfind("PUT /code HTTP/1.1\r\n", 0) == 0);
    REQUIRE(request.substr(request.size() - body.size()) == body);
}

TEST_CASE("CurlHttpClient accepts timeouts too long for a 32-bit long", "[Database][Remote][Curl]")
{
    const NoProxy noProxy;
    OneShotServer server(MakeResponse("200 OK", "{}"));

    // Clamped instead of narrowed: on Windows (32-bit long) a narrowed value would be negative and libcurl would
    // reject every request.
    RemoteDatabaseOptions options;
    options.connectTimeout = std::chrono::milliseconds::max();
    options.requestTimeout = std::chrono::hours(24 * 30);
    CurlHttpClient http(options);

    REQUIRE(http.Send("GET", server.Url() + "/x", {}, "").status == 200);
}

TEST_CASE("CurlHttpClient refuses methods it does not support and a GET with a body", "[Database][Remote][Curl]")
{
    CurlHttpClient http(RemoteDatabaseOptions{});

    // Refused before anything is sent, so no server is needed.
    for (const auto &[method, body] : std::vector<std::pair<std::string, std::string>>{
             {"HEAD", ""}, {"PATCH", "{}"}, {"get", ""}, {"GET", "{}"}})
    {
        CAPTURE(method, body);

        try
        {
            http.Send(method, "http://127.0.0.1:1/x", {}, body);
            FAIL("expected HttpTransportError");
        }
        catch (const HttpTransportError &error)
        {
            REQUIRE_FALSE(error.IsRetryable());
        }
    }
}

TEST_CASE("CurlHttpClient returns error statuses and redirects as responses", "[Database][Remote][Curl]")
{
    const NoProxy noProxy;
    CurlHttpClient http(RemoteDatabaseOptions{});

    SECTION("404 with an error body")
    {
        OneShotServer server(MakeResponse("404 Not Found", R"({"error":"not found"})"));
        const HttpResponse response = http.Send("GET", server.Url() + "/x", {}, "");

        REQUIRE(response.status == 404);
        REQUIRE(response.body == R"({"error":"not found"})");
    }

    SECTION("a redirect is not followed")
    {
        OneShotServer server(MakeResponse("302 Found", "{}", "Location: http://127.0.0.1:1/elsewhere\r\n"));
        const HttpResponse response = http.Send("GET", server.Url() + "/x", {}, "");

        REQUIRE(response.status == 302);
    }
}

TEST_CASE("CurlHttpClient reports transport errors", "[Database][Remote][Curl]")
{
    const NoProxy noProxy;

    SECTION("connection refused is retryable")
    {
        int port = 0;
        close(ListenOnLoopback(port));
        CurlHttpClient http(RemoteDatabaseOptions{});

        try
        {
            http.Send("GET", "http://127.0.0.1:" + std::to_string(port) + "/x", {}, "");
            FAIL("expected HttpTransportError");
        }
        catch (const HttpTransportError &error)
        {
            REQUIRE(error.IsRetryable());
        }
    }

    SECTION("a request timeout is retryable")
    {
        OneShotServer server("");
        RemoteDatabaseOptions options;
        options.requestTimeout = std::chrono::milliseconds(300);
        CurlHttpClient http(options);

        try
        {
            http.Send("POST", server.Url() + "/run", {{"content-type", "application/json"}}, "{}");
            FAIL("expected HttpTransportError");
        }
        catch (const HttpTransportError &error)
        {
            REQUIRE(error.IsRetryable());
        }
    }

    SECTION("another protocol than http and https is rejected and not retryable")
    {
        CurlHttpClient http(RemoteDatabaseOptions{});

        try
        {
            http.Send("GET", "file:///etc/hostname", {}, "");
            FAIL("expected HttpTransportError");
        }
        catch (const HttpTransportError &error)
        {
            REQUIRE_FALSE(error.IsRetryable());
        }
    }
}

#endif // KTT_REMOTE_DATABASE && !_MSC_VER
