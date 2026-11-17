#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/Http/CurlHttpClient.h>
#include <Database/Remote/RemoteDatabase.h>
#include <Database/Remote/Utility/RemoteClient.h>
#include <Database/Remote/Utility/RemoteInternal.h>
#include <Database/Remote/Utility/Wire/Result.h>
#include <Database/Remote/Utility/Wire/Run.h>
#include <Database/Remote/Utility/Wire/Source.h>
#include <Database/Remote/Utility/Wire/Space.h>
#include <Database/Utility/DataDirectory.h>
#include <Database/Utility/TuningInfoValidation.h>
#include <Database/Utility/Uuid.h>

#if !defined(KTT_REMOTE_DATABASE_HOST)
// The ktt-db server of this build. premake sets it from --remote-database-host; the default is a local backend.
#define KTT_REMOTE_DATABASE_HOST "https://localhost:8888"
#endif

namespace ktt::db
{

namespace
{

using json = nlohmann::json;

// The server's maximum for the result limits of POST /result/best and POST .../space/{id}/result.
constexpr uint32_t MaxResultLimit = 1000;

// Page size of GET .../space/{id}/run: the server's maximum, and the batch size of the local Database.
constexpr size_t RunPageSize = 500;

// Longest part of a response body quoted in an exception message.
constexpr size_t MaxQuotedBodyLength = 300;

// A result read from the server, with the kernel duration in nanoseconds the server orders by.
struct RankedResult
{
    int64_t duration;
    KernelResult result;
};

std::vector<RankedResult> ToRankedResults(const wire::ResultList &list)
{
    std::vector<RankedResult> results;
    results.reserve(list.results.size());

    for (const auto &result : list.results)
        results.push_back({result.duration, result.ToKernelResult()});

    return results;
}

std::vector<KernelResult> ToKernelResults(const wire::ResultList &list)
{
    std::vector<KernelResult> results;
    results.reserve(list.results.size());

    for (const auto &result : list.results)
        results.push_back(result.ToKernelResult());

    return results;
}

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

// Sends one request and returns its JSON body. Every request may be retried (RetryPolicy::Idempotent): the reads are
// safe to repeat, and POST /run carries the run id, so the server recognizes a repeated save.
// @return The body of a 200 or 201 response; std::nullopt for 404 when notFoundIsEmpty and the body is a ktt-db error
//         body. Any other 404 (an unmatched route has an empty body, a proxy sends its own page) throws, so a wrong base
//         URL is not mistaken for "not found". If status is not null, it receives the status of a 200 or 201.
// @throw KttException For every other status, a network error, or a success body that is not a JSON object. The
//        message has the operation, the request and the status, never the API key.
std::optional<json> Call(const RemoteClient &client, const std::string &operation, const std::string_view method,
    const std::string &path, const json *body, const bool notFoundIsEmpty, long *status = nullptr)
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

// Carries an exception thrown by a user's predicate to Translate, which rethrows the original exception unchanged.
struct PredicateError
{
    std::exception_ptr error;
};

template <typename Predicate, typename Argument>
bool CallPredicate(const Predicate &predicate, const Argument &argument)
{
    try
    {
        return predicate(argument);
    }
    catch (...)
    {
        throw PredicateError{std::current_exception()};
    }
}

// Runs the body of a public method so that every error names the method: a JSON error (a missing field or a wrong
// type in a response, or request text that is not valid UTF-8) and a KttException from the conversions become a
// KttException that starts with the operation. Other exceptions pass unchanged; a user's predicate is called through
// CallPredicate, so its exceptions, a KttException included, reach the caller unchanged.
template <typename Function>
auto Translate(const std::string &operation, const RemoteClient &client, Function function) -> decltype(function())
{
    try
    {
        return function();
    }
    catch (const PredicateError &error)
    {
        std::rethrow_exception(error.error);
    }
    catch (const json::exception &error)
    {
        throw KttException(operation + " failed: invalid JSON: " + client.Redact(error.what()));
    }
    catch (const KttException &error)
    {
        const std::string message = error.what();

        if (message.rfind(operation, 0) == 0)
            throw;

        throw KttException(operation + " failed: " + client.Redact(message), error.GetReason());
    }
}

// Name of the API key file in KTT's data directory (next to the default local database)
constexpr const char *ApiKeyFileName = "api-key";

// A key is a short token; anything larger is not a key file.
constexpr std::uintmax_t MaxApiKeyFileSize = 4096;

// Reads the API key from a key file, without surrounding whitespace. The messages name the file, never its content.
std::string ReadApiKeyFile(const std::filesystem::path &path)
{
    const std::string where = "RemoteDatabase: the API key file " + path.u8string();
    std::error_code error;

    if (!std::filesystem::exists(path, error))
    {
        throw KttException(where + " does not exist; write the key to it or pass the key to the RemoteDatabase "
            "constructor");
    }

    if (!std::filesystem::is_regular_file(path, error))
        throw KttException(where + " is not a file");

    const std::uintmax_t size = std::filesystem::file_size(path, error);

    if (error || size > MaxApiKeyFileSize)
        throw KttException(where + (error ? " cannot be read" : " is too large to hold an API key"));

    std::ifstream file(path, std::ios::binary);
    std::string text(static_cast<size_t>(size), '\0');

    if (!file || !file.read(text.data(), static_cast<std::streamsize>(text.size())))
        throw KttException(where + " cannot be read");

    static constexpr const char *Whitespace = " \t\r\n";
    const size_t first = text.find_first_not_of(Whitespace);

    if (first == std::string::npos)
        throw KttException(where + " is empty");

    const size_t last = text.find_last_not_of(Whitespace);
    return text.substr(first, last - first + 1);
}

void CheckLimit(const std::string &operation, const uint32_t limit)
{
    if (limit > MaxResultLimit)
    {
        throw KttException(operation + ": the limit must be at most " + std::to_string(MaxResultLimit)
            + " for a remote database, got " + std::to_string(limit));
    }
}

} // namespace

RemoteDatabase::RemoteDatabase(RemoteDatabaseOptions options) :
    RemoteDatabase(ReadApiKeyFile(GetDefaultApiKeyPath()), std::move(options))
{}

RemoteDatabase::RemoteDatabase(std::string apiKey, RemoteDatabaseOptions options) :
    RemoteDatabase(std::move(apiKey), options, std::make_shared<CurlHttpClient>(options))
{}

RemoteDatabase::RemoteDatabase(RemoteDatabaseOptions options, std::shared_ptr<HttpClient> httpClient) :
    RemoteDatabase(ReadApiKeyFile(GetDefaultApiKeyPath()), std::move(options), std::move(httpClient))
{}

RemoteDatabase::RemoteDatabase(std::string apiKey, RemoteDatabaseOptions options,
    std::shared_ptr<HttpClient> httpClient) :
    m_Client(std::make_unique<RemoteClient>(GetHost(), std::move(apiKey), options.maxRetries, std::move(httpClient))),
    m_SourceCodeProvider(std::move(options.sourceCodeProvider))
{}

std::string RemoteDatabase::GetHost()
{
    std::string host = KTT_REMOTE_DATABASE_HOST;

    while (!host.empty() && host.back() == '/')
        host.pop_back();

    return host;
}

std::filesystem::path RemoteDatabase::GetDefaultApiKeyPath()
{
    try
    {
        return GetDefaultDataDirectory() / ApiKeyFileName;
    }
    catch (const KttException &exception)
    {
        throw KttException(std::string(exception.what()) + "; pass the API key to the RemoteDatabase constructor");
    }
}

RemoteDatabase::~RemoteDatabase() = default;

void RemoteDatabase::SaveResults(const TuningInfo &source, std::vector<KernelResult> results,
    const SaveOptions options) const
{
    static const std::string Operation = "RemoteDatabase::SaveResults";
    ValidateTuningInfo(source);

    const auto response = Translate(Operation, *m_Client, [&]
    {
        // Generated once, before the request: every retry sends the same id, so the server saves the run only once.
        const json body = wire::SaveRunRequest::From(UuidGenerator::GenerateUuidV7(), source, results, options.format);

        // Without notFoundIsEmpty, Call returns a body or throws.
        return Call(*m_Client, Operation, "POST", "/run", &body, false)->get<wire::SaveRunResponse>();
    });

    UploadMissingSourceCode(*this, m_SourceCodeProvider, source.spaceInfo.sourceFingerprint, response.sourceHasCode);
}

std::vector<KernelResult> RemoteDatabase::SimpleGetBestResults(const TuningInfo &source, const uint32_t limit) const
{
    static const std::string Operation = "RemoteDatabase::SimpleGetBestResults";
    ValidateTuningInfo(source);

    if (limit == 0)
        return {};

    CheckLimit(Operation, limit);

    return Translate(Operation, *m_Client, [&]() -> std::vector<KernelResult>
    {
        const json body = wire::BestResultsRequest{wire::TuningInfo::From(source), limit};
        // The endpoint answers an unknown source or space with an empty list, so a 404 is an error here.
        const auto response = Call(*m_Client, Operation, "POST", "/result/best", &body, false);
        return ToKernelResults(response->get<wire::ResultList>());
    });
}

std::vector<KernelResult> RemoteDatabase::GetBestResults(const GetBestResultsQuery &query) const
{
    static const std::string Operation = "RemoteDatabase::GetBestResults";

    if (query.limit == 0)
        return {};

    CheckLimit(Operation, query.limit);

    return Translate(Operation, *m_Client, [&]() -> std::vector<KernelResult>
    {
        const std::string sourcePath = "/source/" + wire::FingerprintToString(query.source.sourceFingerprint);
        const auto space = Call(*m_Client, Operation, "GET", sourcePath + "/space?parameter_fingerprint="
            + wire::FingerprintToString(query.source.parameterFingerprint) + "&space_fingerprint="
            + wire::FingerprintToString(query.source.spaceFingerprint), nullptr, true);

        if (!space)
            return {};

        const std::string spacePath =
            sourcePath + "/space/" + std::to_string(space->get<wire::FindSpaceResponse>().spaceId);

        // Offset paging shifts when a run becomes visible meanwhile, so a run can come back on a later page.
        std::set<uuid> seenRuns;
        std::vector<RankedResult> bestResults;

        for (size_t offset = 0;;)
        {
            const auto page = Call(*m_Client, Operation, "GET", spacePath + "/run?offset=" + std::to_string(offset)
                + "&limit=" + std::to_string(RunPageSize), nullptr, true);

            if (!page)
                return {};

            const auto runs = page->get<wire::SpaceRunsResponse>().runs;
            wire::RunResultsRequest request{{}, query.limit};

            for (const auto &run : runs)
            {
                if (!seenRuns.insert(run.runId).second)
                    continue;

                bool keep = true;

                if (query.devicePredicate)
                    keep = CallPredicate(*query.devicePredicate, run.device.ToDeviceInfo());

                // As in the local Database: with an input predicate, a run without input data is left out.
                if (keep && query.inputPredicate)
                    keep = run.inputData.has_value() && CallPredicate(*query.inputPredicate, *run.inputData);

                if (keep)
                    request.runIds.push_back(run.runId);
            }

            if (!request.runIds.empty())
            {
                const json body = request;
                const auto response = Call(*m_Client, Operation, "POST", spacePath + "/result", &body, true);

                if (!response)
                    return {};

                auto batch = ToRankedResults(response->get<wire::ResultList>());

                // Both ranges are sorted by kernel duration: bestResults by earlier merges, the batch by the server.
                // The merge is stable, so equal durations keep their order and earlier pages win ties, as locally.
                const auto oldSize = static_cast<std::ptrdiff_t>(bestResults.size());
                bestResults.insert(bestResults.end(), std::make_move_iterator(batch.begin()),
                    std::make_move_iterator(batch.end()));
                std::inplace_merge(bestResults.begin(), bestResults.begin() + oldSize, bestResults.end(),
                    [](const RankedResult &left, const RankedResult &right) { return left.duration < right.duration; });

                if (bestResults.size() > query.limit)
                    bestResults.erase(bestResults.begin() + query.limit, bestResults.end());
            }

            if (runs.size() < RunPageSize)
                break;

            offset += runs.size();
        }

        std::vector<KernelResult> kernelResults;
        kernelResults.reserve(bestResults.size());

        for (auto &result : bestResults)
            kernelResults.push_back(std::move(result.result));

        return kernelResults;
    });
}

std::optional<SourceStats> RemoteDatabase::GetStatsForSource(const size_t sourceFingerprint) const
{
    static const std::string Operation = "RemoteDatabase::GetStatsForSource";

    return Translate(Operation, *m_Client, [&]() -> std::optional<SourceStats>
    {
        const auto response = Call(*m_Client, Operation, "GET",
            "/source/" + wire::FingerprintToString(sourceFingerprint) + "/stats", nullptr, true);

        if (!response)
            return std::nullopt;

        return response->get<wire::SourceStatsResponse>().ToSourceStats();
    });
}

// The helpers above, for the other source files of RemoteDatabase (Remote/Utility/RemoteInternal.h).

std::optional<json> CallRemote(const RemoteClient &client, const std::string &operation, const std::string_view method,
    const std::string &path, const json *body, const bool notFoundIsEmpty, long *status)
{
    return Call(client, operation, method, path, body, notFoundIsEmpty, status);
}

void TranslateRemote(const std::string &operation, const RemoteClient &client, const std::function<void()> &function)
{
    Translate(operation, client, function);
}

} // namespace ktt::db
