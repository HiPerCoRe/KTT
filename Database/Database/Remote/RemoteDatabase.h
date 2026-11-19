#pragma once
#include <KttPlatform.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Database/IDatabase.h>
#include <Database/Remote/ResultQuery.h>
#include <KttTypes.h>

namespace ktt
{
class Tuner;
} // namespace ktt

namespace ktt::db
{

class HttpClient;
class RemoteClient;

/** @typedef SourceCodeProvider
 * Returns the kernel definitions whose source fingerprint is sourceFingerprint, in the order of the kernel's
 * definitions (Tuner::GetKernelSources), or std::nullopt when it does not know that source. Used by
 * RemoteDatabase::SaveResults to upload the source code the server does not have yet (see
 * RemoteDatabaseOptions::sourceCodeProvider). It is called on the thread that calls SaveResults, so from several
 * threads at once when SaveResults is.
 */
using SourceCodeProvider = std::function<std::optional<std::vector<SourceDefinition>>(size_t sourceFingerprint)>;

/** @struct RemoteDatabaseOptions
 * Connection settings of a RemoteDatabase. A proxy is taken from the environment variables that libcurl reads:
 * http_proxy (lower case only), and https_proxy, all_proxy and no_proxy (in lower or upper case).
 */
struct RemoteDatabaseOptions
{
    std::chrono::milliseconds connectTimeout{5000}; ///< Time limit for establishing the connection (0 = none).
    std::chrono::milliseconds requestTimeout{60000}; ///< Time limit for a whole request, per attempt (0 = none).
    uint32_t maxRetries{3}; ///< Additional attempts after a connection error, a timeout or 502 / 503 / 504.
    bool verifyTls{true}; ///< Verify the server's TLS certificate and host name. Disable only for testing.
    std::optional<std::filesystem::path> caBundle; ///< A custom CA bundle (PEM), e.g. for an internal server.
    /// Supplies the kernel source code after a save. When it is set and the server answers a SaveResults with
    /// source_has_code false, the client asks it for the code of the saved source and uploads it with
    /// RemoteDatabase::SaveSourceCode. A failed upload only logs a warning. Empty (the default): nothing is uploaded.
    /// See MakeSourceCodeProvider for a provider that serves the kernels of a Tuner.
    SourceCodeProvider sourceCodeProvider;
};

/** @class RemoteDatabase
 * Stores and reads tuning results on a ktt-db server through its application API (HTTP, authenticated with an API
 * key), as an alternative to the local SQLite Database. Available when KTT is built with --remote-database.
 * The API key decides which realms are read and written; there is no realm parameter. Every failure throws
 * KttException, and the API key never appears in exception messages or logs.
 * Requests are retried with exponential backoff after connection errors, timeouts and the statuses 502, 503 and 504
 * (see RemoteDatabaseOptions::maxRetries). The methods are const and can be called from several threads at once.
 * Results are read in the time unit their run was saved with by switching KTT's process-wide time unit for a moment
 * (see TimeUnitGuard); other code that reads or sets the time unit at the same time can see the switched unit.
 */
class KTT_API RemoteDatabase : public IDatabase
{
public:
    /** @fn RemoteDatabase(RemoteDatabaseOptions options = {})
     * Creates a client for the ktt-db server (see GetHost) that reads its API key from the key file (see
     * GetDefaultApiKeyPath): the file holds only the key; whitespace and line breaks around it are ignored. The
     * constructor does not contact the server.
     * @param options Timeouts, retries and TLS settings.
     * @throw KttException If the key file location cannot be determined, the file does not exist or cannot be read,
     *        or the key in it is empty or contains a control character. The message names the file, never the key.
     */
    explicit RemoteDatabase(RemoteDatabaseOptions options = {});

    /** @fn RemoteDatabase(std::string apiKey, RemoteDatabaseOptions options = {})
     * Creates a client for the ktt-db server (see GetHost) with the API key given by the application. The key file is
     * not read. The constructor does not contact the server.
     * @param apiKey API key created in the web app's settings (sent as x-api-key).
     * @param options Timeouts, retries and TLS settings.
     * @throw KttException If apiKey is empty or contains a control character.
     */
    explicit RemoteDatabase(std::string apiKey, RemoteDatabaseOptions options = {});

    /** @fn RemoteDatabase(std::string apiKey, RemoteDatabaseOptions options, std::shared_ptr<HttpClient> httpClient)
     * Creates a client that sends its requests through the given HTTP client instead of libcurl. HttpClient is an
     * internal type, so this constructor is meant for tests. The timeouts and TLS settings in options are not used.
     * @param apiKey API key sent with every request.
     * @param options Retry settings.
     * @param httpClient The HTTP client. Must not be null.
     * @throw KttException If apiKey is invalid or httpClient is null.
     */
    RemoteDatabase(std::string apiKey, RemoteDatabaseOptions options, std::shared_ptr<HttpClient> httpClient);

    /** @fn RemoteDatabase(RemoteDatabaseOptions options, std::shared_ptr<HttpClient> httpClient)
     * Like the previous constructor, but reads the API key from the key file (see GetDefaultApiKeyPath). Meant for
     * tests.
     * @param options Retry settings.
     * @param httpClient The HTTP client. Must not be null.
     * @throw KttException If the key file cannot be read or holds an invalid key, or httpClient is null.
     */
    RemoteDatabase(RemoteDatabaseOptions options, std::shared_ptr<HttpClient> httpClient);

    /** @fn static std::string GetHost()
     * Returns the ktt-db server every RemoteDatabase talks to, e.g. https://supercuda.fi.muni.cz:8443. It is fixed when KTT is
     * built (premake option --remote-database-host) and cannot be changed by the application. The client appends
     * /ktt-db/app/... to it.
     * @return The server root including the scheme, without a trailing slash.
     */
    static std::string GetHost();

    /** @fn static std::filesystem::path GetDefaultApiKeyPath()
     * Returns the API key file read by the constructors without an apiKey parameter: the file api-key in the same
     * directory as the default local database, so %LOCALAPPDATA%\ktt\api-key on Windows and $XDG_DATA_HOME/ktt/api-key
     * (falling back to ~/.local/share/ktt/api-key) elsewhere. Does not check that the file exists.
     * @return Path of the key file.
     * @throw KttException If the location cannot be determined (the environment variables above are unset).
     */
    static std::filesystem::path GetDefaultApiKeyPath();

    /** @fn ~RemoteDatabase()
     * Destructor.
     */
    ~RemoteDatabase() override;

    // Disable copy
    RemoteDatabase(const RemoteDatabase &) = delete;
    RemoteDatabase &operator=(const RemoteDatabase &) = delete;

    /** @fn void SaveResults(const TuningInfo& source, std::vector<KernelResult> results, SaveOptions options = {}) const
     * Saves the kernel results of one tuning run on the server (POST /ktt-db/app/run) and links the run to every
     * usable realm of the API key. As locally, only results with status Ok are saved, and a run without results is
     * still created. The run gets a UUIDv7 id generated before the request, so a retried request cannot store the run
     * twice. The payloads are written in the current time unit, which the run records.
     * Differences from Database: OutputFormat::XML is stored as JSON (a warning is logged once per process), and
     * options.indent is ignored. The source and space names are sent with every save, but the server applies them only
     * when the source or space first appears in a realm; it trims them, and a name longer than 100 characters is
     * rejected.
     * @param source The TuningInfo containing source fingerprint, space info, device info, and input data.
     * @param results Kernel results of the run.
     * @param options Output format; indent is not used.
     * @throw KttException If source is invalid (before any request), the server rejects the run (e.g. HTTP 400, 401,
     *        403, 409 or 413), or the request fails after its retries.
     */
    void SaveResults(const TuningInfo &source, std::vector<KernelResult> results,
        SaveOptions options = {}) const override;

    /** @fn std::vector<KernelResult> SimpleGetBestResults(const TuningInfo& source, uint32_t limit = 50) const
     * Retrieves the best stored results for the tuning problem on a matching device (POST /ktt-db/app/result/best):
     * the same device model, or for CUDA the same compute capability major and a minor within +-1, as locally. Only
     * runs the API key can see are searched.
     * @param source The TuningInfo containing source fingerprint, space info, and device info to search for.
     * @param limit Maximum number of results to return, at most 1000. 0 returns no results without a request.
     * @return Results ordered by the stored kernel duration, fastest first; empty for an unknown source or space.
     * @throw KttException If source is invalid, limit is above 1000, or the request fails.
     */
    std::vector<KernelResult> SimpleGetBestResults(const TuningInfo &source, uint32_t limit = 50) const override;

    /** @fn std::vector<KernelResult> GetBestResults(const GetBestResultsQuery& query) const
     * Retrieves the best stored results of the runs selected by the query's predicates. The predicates run on the
     * client, so this takes several requests: the space is looked up, its runs are read in pages of 500, and the
     * results of the selected runs of each page are fetched and merged by kernel duration. As locally, a run without
     * input data is left out when an input predicate is set. A run listed twice (offset paging can shift while runs
     * are being saved) is used once. Only runs the API key can see are searched.
     * @param query Source info, optional device and input-data predicates, and a result limit of at most 1000.
     * @return Results ordered by the stored kernel duration, fastest first, limited to query.limit; empty for an
     *         unknown source or space and for a limit of 0.
     * @throw KttException If query.limit is above 1000 or a request fails. An exception thrown by a predicate
     *        propagates.
     */
    std::vector<KernelResult> GetBestResults(const GetBestResultsQuery &query) const override;

    /** @fn std::optional<SourceStats> GetStatsForSource(size_t sourceFingerprint) const
     * Retrieves aggregate statistics for a tuning source (GET /ktt-db/app/source/{fingerprint}/stats). The counts
     * cover only the runs the API key can see, so they can be lower than the server's totals, and name is always
     * empty: the application API returns no names.
     * @param sourceFingerprint The fingerprint hash identifying the tuning source.
     * @return The statistics, or std::nullopt if the source is not found or the key sees none of its runs.
     * @throw KttException If the request fails.
     */
    std::optional<SourceStats> GetStatsForSource(size_t sourceFingerprint) const override;

    /** @fn std::vector<ResultQueryItem> QueryResults(const ResultQuery& query) const
     * Searches every result the API key can see with an OData-style filter (POST /ktt-db/app/result/query). Remote
     * only: the local Database has no equivalent. The filter is sent unchanged and checked by the server, whose
     * message points at the problem. Unlike GetBestResults, the condition runs on the server in one request, so it is
     * the fast path whenever it can be written as a filter. Page with query.top and query.skip.
     * A query that runs into the server's statement timeout (HTTP 503 with code query_timeout) is not retried,
     * because it would time out again; narrow the filter instead. Other 502 / 503 / 504 responses are retried.
     * @param query Filter, order, page and whether to include the stored payloads.
     * @return One page of results in the requested order; empty when nothing matches or query.top is 0.
     * @throw KttException If query.top is above 1000 or query.skip above 100000 (before any request), the server
     *        rejects the query (e.g. HTTP 400 for a bad filter, with the server's message), the query times out, or
     *        the request fails after its retries.
     */
    std::vector<ResultQueryItem> QueryResults(const ResultQuery &query) const;

    /** @fn bool SaveSourceCode(size_t sourceFingerprint, const std::vector<SourceDefinition>& definitions) const
     * Uploads the source code of a kernel (PUT /ktt-db/app/source/{fingerprint}/code), so that the web app can show
     * it. Remote only: the local Database stores no source code. The server keeps the code once per source and never
     * changes it, and it accepts it only if the sources have the given fingerprint. The source must already be on the
     * server and visible to the API key, i.e. a run of it must have been saved first. SaveResults calls this by
     * itself when RemoteDatabaseOptions::sourceCodeProvider is set.
     * Before the request the client checks what the server would reject: 1 to 32 definitions, each source at most
     * 1 MiB and all of them at most 4 MiB, each name at most 1 KiB (UTF-8 bytes), valid UTF-8 without U+0000, a
     * request body of at most 5 MiB, and the source fingerprint of the definitions, recomputed as KTT computes it.
     * @param sourceFingerprint The source fingerprint (TuningSpaceInfo::sourceFingerprint) the code belongs to.
     * @param definitions The kernel's definitions in the order of its definitions, e.g. from
     *        Tuner::GetKernelSources. The sources are sent byte for byte, with nothing trimmed or converted; the names
     *        are informational and not part of the fingerprint.
     * @return True if the code was stored now (HTTP 201), false if the server already had it (HTTP 200; also the
     *         case when an earlier attempt of the same call stored it and its answer was lost).
     * @throw KttException If a check above fails (before any request; a fingerprint mismatch means the definitions are
     *        not the ones the fingerprint was computed from), the source is unknown or not visible to the key (HTTP
     *        404), the server rejects the code (e.g. HTTP 400, 403 or 413), or the request fails after its retries.
     */
    bool SaveSourceCode(size_t sourceFingerprint, const std::vector<SourceDefinition> &definitions) const;

private:
    std::unique_ptr<RemoteClient> m_Client;
    SourceCodeProvider m_SourceCodeProvider;
};

/** @fn SourceCodeProvider MakeSourceCodeProvider(const Tuner& tuner, std::vector<KernelId> kernels)
 * Creates a source-code provider (RemoteDatabaseOptions::sourceCodeProvider) that serves the given kernels of a tuner:
 * for a source fingerprint it returns Tuner::GetKernelSources of the first of the kernels whose sources have that
 * fingerprint, and std::nullopt if none has. The provider keeps a reference to the tuner, which must outlive every
 * RemoteDatabase that uses it, and it runs on the thread that calls SaveResults.
 * @param tuner The tuner that holds the kernels.
 * @param kernels Ids of the kernels whose results are saved through the RemoteDatabase.
 * @return The provider.
 */
KTT_API SourceCodeProvider MakeSourceCodeProvider(const Tuner &tuner, std::vector<KernelId> kernels);

} // namespace ktt::db
