#include <string>

#include <Database/Remote/Utility/RemoteCall.h>

namespace ktt::db
{

namespace
{

using json = nlohmann::json;

// Longest part of a response body quoted in an exception message.
constexpr size_t MaxQuotedBodyLength = 300;

std::string Quote(const std::string &text)
{
    if (text.size() <= MaxQuotedBodyLength)
        return text;

    return text.substr(0, MaxQuotedBodyLength) + "...";
}

// Whether a body is a ktt-db error body: a JSON object with a string "error" (endpoints.md, "Errors").
bool IsErrorBody(const std::string &body)
{
    const json parsed = json::parse(body, nullptr, false);
    return parsed.is_object() && parsed.contains("error") && parsed["error"].is_string();
}

// The server's explanation of an error response: the "error" (and "code") of an ApiErrorBody, or the start of the
// body when it is something else, e.g. the HTML page of a proxy.
std::string DescribeErrorBody(const std::string &body)
{
    const json parsed = json::parse(body, nullptr, false);

    if (parsed.is_object() && parsed.contains("error") && parsed["error"].is_string())
    {
        std::string text = parsed["error"].get<std::string>();

        if (parsed.contains("code") && parsed["code"].is_string())
            text += " (code " + parsed["code"].get<std::string>() + ")";

        return Quote(text);
    }

    if (body.empty())
        return "empty response body";

    return "response body is not a ktt-db error: " + Quote(body);
}

} // namespace

std::optional<json> CallRemote(const RemoteClient &client, const std::string &operation, const std::string_view method,
    const std::string &path, const json *body, const bool notFoundIsEmpty, long *status)
{
    HttpResponse response;

    try
    {
        response = client.Send(method, path, body != nullptr ? body->dump() : std::string(), RetryPolicy::Idempotent);
    }
    catch (const KttException &error)
    {
        throw KttException(operation + " failed: " + client.Redact(error.what()));
    }

    if (response.status == 404 && notFoundIsEmpty && IsErrorBody(response.body))
        return std::nullopt;

    const std::string request = std::string(method) + " " + path;

    if (response.status != 200 && response.status != 201)
    {
        throw KttException(operation + " failed: HTTP " + std::to_string(response.status) + " from " + request + ": "
            + DescribeErrorBody(client.Redact(response.body)));
    }

    json parsed = json::parse(response.body, nullptr, false);

    if (!parsed.is_object())
    {
        throw KttException(operation + " failed: the body of HTTP " + std::to_string(response.status) + " from "
            + request + " is not a JSON object: " + Quote(client.Redact(response.body)));
    }

    if (status != nullptr)
        *status = response.status;

    return parsed;
}

} // namespace ktt::db
