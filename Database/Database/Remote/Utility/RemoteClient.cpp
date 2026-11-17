#include <algorithm>
#include <cctype>
#include <cstdint>
#include <thread>
#include <utility>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/Utility/RemoteClient.h>
#include <Database/Utility/Uuid.h>

namespace ktt::db
{

static bool StartsWithIgnoringCase(const std::string &text, std::string_view prefix)
{
    return text.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), text.begin(), [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
    });
}

static std::string NormalizeBaseUrl(std::string baseUrl)
{
    const bool http = StartsWithIgnoringCase(baseUrl, "http://");
    const bool https = StartsWithIgnoringCase(baseUrl, "https://");

    while (!baseUrl.empty() && baseUrl.back() == '/')
        baseUrl.pop_back();

    const size_t schemeLength = https ? 8 : 7;

    // The messages leave the URL out: swapped constructor arguments would put the API key there, and a URL can carry
    // credentials (user:password@host) or a token in its query.
    if ((!http && !https) || baseUrl.size() <= schemeLength)
        throw KttException("The remote database URL must be an http:// or https:// URL with a host");

    const bool hasInvalidCharacter = std::any_of(baseUrl.begin(), baseUrl.end(), [](char c) {
        return std::isspace(static_cast<unsigned char>(c)) || std::iscntrl(static_cast<unsigned char>(c)) || c == '?'
            || c == '#';
    });

    if (hasInvalidCharacter)
        throw KttException("The remote database URL must not contain whitespace, a query or a fragment");

    return baseUrl;
}

static void ValidateApiKey(const std::string &apiKey)
{
    // The key is never part of the message.
    if (apiKey.empty())
        throw KttException("The remote database API key must not be empty");

    const bool hasControlCharacter = std::any_of(apiKey.begin(), apiKey.end(), [](char c) {
        return std::iscntrl(static_cast<unsigned char>(c));
    });

    if (hasControlCharacter)
        throw KttException("The remote database API key must not contain line breaks or other control characters");
}

RemoteClient::RemoteClient(std::string baseUrl, std::string apiKey, const uint32_t maxRetries,
    std::shared_ptr<HttpClient> httpClient, Sleeper sleeper) :
    m_BaseUrl(NormalizeBaseUrl(std::move(baseUrl))),
    m_ApiKey(std::move(apiKey)),
    m_MaxRetries(maxRetries),
    m_HttpClient(std::move(httpClient)),
    m_Sleeper(std::move(sleeper))
{
    ValidateApiKey(m_ApiKey);

    if (m_HttpClient == nullptr)
        throw KttException("The remote database HTTP client must not be null");

    if (!m_Sleeper)
        m_Sleeper = [](const std::chrono::milliseconds duration) { std::this_thread::sleep_for(duration); };
}

HttpResponse RemoteClient::Send(const std::string_view method, const std::string_view path, const std::string &body,
    const RetryPolicy retry) const
{
    const std::string url = m_BaseUrl + "/ktt-db/app" + std::string(path);
    const uint64_t maxAttempts = retry == RetryPolicy::Idempotent ? uint64_t{m_MaxRetries} + 1 : 1;

    // One correlation id per logical request: the retries reuse it, so the server logs tie the attempts together.
    HttpHeaders headers{
        {"x-api-key", m_ApiKey},
        {"correlation-id", UuidGenerator::ToString(UuidGenerator::GenerateUuid())},
        {"accept", "application/json"}};

    if (!body.empty())
        headers.emplace_back("content-type", "application/json");

    // 64 bits, so that maxRetries == UINT32_MAX (2^32 attempts) still ends.
    for (uint64_t attempt = 1;; ++attempt)
    {
        const bool lastAttempt = attempt >= maxAttempts;

        try
        {
            HttpResponse response = m_HttpClient->Send(method, url, headers, body);

            if (lastAttempt || !IsRetryableResponse(response))
                return response;
        }
        catch (const HttpTransportError &error)
        {
            if (lastAttempt || !error.IsRetryable())
            {
                const std::string attempts = attempt > 1 ? " (after " + std::to_string(attempt) + " attempts)" : "";
                throw KttException("ktt-db request " + std::string(method) + " " + std::string(path) + " failed"
                    + attempts + ": " + Redact(error.what()));
            }
        }

        m_Sleeper(GetBackoff(static_cast<uint32_t>(std::min<uint64_t>(attempt, UINT32_MAX))));
    }
}

std::string RemoteClient::Redact(std::string text) const
{
    static const std::string Replacement = "<api key>";

    for (size_t position = text.find(m_ApiKey); position != std::string::npos;
         position = text.find(m_ApiKey, position + Replacement.size()))
        text.replace(position, m_ApiKey.size(), Replacement);

    return text;
}

const std::string &RemoteClient::GetBaseUrl() const
{
    return m_BaseUrl;
}

std::chrono::milliseconds RemoteClient::GetBackoff(const uint32_t retry)
{
    std::chrono::milliseconds backoff = InitialBackoff;

    for (uint32_t i = 1; i < retry && backoff < MaxBackoff; ++i)
        backoff *= 2;

    return std::min(backoff, MaxBackoff);
}

bool RemoteClient::IsRetryableResponse(const HttpResponse &response)
{
    if (response.status != 502 && response.status != 503 && response.status != 504)
        return false;

    if (response.status == 503)
    {
        const nlohmann::json body = nlohmann::json::parse(response.body, nullptr, false);

        if (body.is_object() && body.contains("code") && body["code"] == "query_timeout")
            return false;
    }

    return true;
}

} // namespace ktt::db
