#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <set>
#include <string>
#include <utility>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/Http/CurlHttpClient.h>
#include <Database/Remote/RemoteDatabase.h>
#include <Database/Remote/Utility/ApiKeyFile.h>
#include <Database/Remote/Utility/RemoteCall.h>
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
// The ktt-db server of this build. premake sets it from --remote-database-host; the default is the FI MUNI server (needs the FI MUNI VPN).
#define KTT_REMOTE_DATABASE_HOST "https://supercuda.fi.muni.cz:8443"
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

    const auto response = TranslateRemote(Operation, *m_Client, [&]
    {
        // Generated once, before the request: every retry sends the same id, so the server saves the run only once.
        const json body = wire::SaveRunRequest::From(UuidGenerator::GenerateUuidV7(), source, results, options.format);

        // Without notFoundIsEmpty, CallRemote returns a body or throws.
        return CallRemote(*m_Client, Operation, "POST", "/run", &body, false)->get<wire::SaveRunResponse>();
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

    return TranslateRemote(Operation, *m_Client, [&]() -> std::vector<KernelResult>
    {
        const json body = wire::BestResultsRequest{wire::TuningInfo::From(source), limit};
        // The endpoint answers an unknown source or space with an empty list, so a 404 is an error here.
        const auto response = CallRemote(*m_Client, Operation, "POST", "/result/best", &body, false);
        return ToKernelResults(response->get<wire::ResultList>());
    });
}

std::vector<KernelResult> RemoteDatabase::GetBestResults(const GetBestResultsQuery &query) const
{
    static const std::string Operation = "RemoteDatabase::GetBestResults";

    if (query.limit == 0)
        return {};

    CheckLimit(Operation, query.limit);

    return TranslateRemote(Operation, *m_Client, [&]() -> std::vector<KernelResult>
    {
        const std::string sourcePath = "/source/" + wire::FingerprintToString(query.source.sourceFingerprint);
        const auto space = CallRemote(*m_Client, Operation, "GET", sourcePath + "/space?parameter_fingerprint="
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
            const auto page = CallRemote(*m_Client, Operation, "GET", spacePath + "/run?offset=" + std::to_string(offset)
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
                const auto response = CallRemote(*m_Client, Operation, "POST", spacePath + "/result", &body, true);

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

    return TranslateRemote(Operation, *m_Client, [&]() -> std::optional<SourceStats>
    {
        const auto response = CallRemote(*m_Client, Operation, "GET",
            "/source/" + wire::FingerprintToString(sourceFingerprint) + "/stats", nullptr, true);

        if (!response)
            return std::nullopt;

        return response->get<wire::SourceStatsResponse>().ToSourceStats();
    });
}

} // namespace ktt::db
