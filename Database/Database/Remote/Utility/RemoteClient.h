#pragma once

// Internal to the remote database client (not exported).

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include <Database/Remote/Http/HttpClient.h>

namespace ktt::db
{

/** @enum RetryPolicy
 * Whether a request may be sent again after a connection error, a timeout or 502 / 503 / 504.
 */
enum class RetryPolicy
{
    Idempotent, ///< The request is safe to repeat (all GETs, the read-only POSTs, POST /run, PUT .../code).
    Never ///< The request is sent once.
};

/** @class RemoteClient
 * Sends requests to the ktt-db application API (<baseUrl>/ktt-db/app/...) through an HttpClient: adds the API key,
 * a correlation id (one per Send, kept across its retries) and the content type, and retries idempotent requests with
 * exponential backoff. It does not interpret the status codes beyond the retry decision; the caller maps them to
 * results or exceptions.
 * Send is const and safe to call from several threads at once when the HttpClient is.
 */
class RemoteClient
{
public:
    /** Waits between two attempts. Replaced in tests to avoid real sleeps. */
    using Sleeper = std::function<void(std::chrono::milliseconds)>;

    /** Wait before the first retry; it doubles with each further retry, up to MaxBackoff. */
    static constexpr std::chrono::milliseconds InitialBackoff{250};
    static constexpr std::chrono::milliseconds MaxBackoff{8000};

    /** @fn RemoteClient(std::string baseUrl, std::string apiKey, uint32_t maxRetries, std::shared_ptr<HttpClient>
     * httpClient, Sleeper sleeper = {})
     * @param baseUrl Server root, http:// or https://. A trailing slash is removed.
     * @param apiKey API key, sent as x-api-key.
     * @param maxRetries Additional attempts for RetryPolicy::Idempotent requests.
     * @param httpClient Sends the requests. Must not be null.
     * @param sleeper Waits between attempts; std::this_thread::sleep_for when empty.
     * @throw KttException If baseUrl or apiKey is invalid, or httpClient is null. The message has neither the key
     *        nor the URL.
     */
    RemoteClient(std::string baseUrl, std::string apiKey, uint32_t maxRetries, std::shared_ptr<HttpClient> httpClient,
        Sleeper sleeper = {});

    /** @fn HttpResponse Send(std::string_view method, std::string_view path, const std::string& body,
     * RetryPolicy retry) const
     * Sends a request to <baseUrl>/ktt-db/app<path>, retrying it per the policy. Every call gets a new
     * correlation id, which all of its attempts share. When the retries run out on a 502 / 503 / 504, that last
     * response is returned.
     * @param method HTTP method.
     * @param path Path below /ktt-db/app, starting with '/'. Path segments must already be URL-safe.
     * @param body JSON body; empty for none (then no content-type is sent).
     * @param retry Whether the request may be repeated.
     * @return The response of the last attempt.
     * @throw KttException If no response was received (after the retries); the message has the method, the path and
     *        the transport error.
     */
    HttpResponse Send(std::string_view method, std::string_view path, const std::string &body,
        RetryPolicy retry) const;

    /** @fn std::string Redact(std::string text) const
     * Replaces every occurrence of the API key in text with "<api key>". Applied to every text that goes into an
     * exception message, as a safeguard in case a server or proxy echoes the key back.
     * @param text Text to clean.
     * @return The text without the API key.
     */
    std::string Redact(std::string text) const;

    /** @fn const std::string& GetBaseUrl() const
     * @return The server root without a trailing slash.
     */
    const std::string &GetBaseUrl() const;

    /** @fn static std::chrono::milliseconds GetBackoff(uint32_t retry)
     * @param retry Number of the retry, starting at 1.
     * @return The wait before that retry.
     */
    static std::chrono::milliseconds GetBackoff(uint32_t retry);

    /** @fn static bool IsRetryableResponse(const HttpResponse& response)
     * @return True for 502, 503 and 504, except a 503 whose body has "code": "query_timeout" (the query itself
     *         is too slow, so repeating it would only time out again).
     */
    static bool IsRetryableResponse(const HttpResponse &response);

private:
    std::string m_BaseUrl;
    std::string m_ApiKey;
    uint32_t m_MaxRetries;
    std::shared_ptr<HttpClient> m_HttpClient;
    Sleeper m_Sleeper;
};

} // namespace ktt::db
