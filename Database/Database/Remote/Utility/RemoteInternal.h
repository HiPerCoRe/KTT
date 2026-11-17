#pragma once

// Internal to the remote database client (not exported): what the source files of RemoteDatabase share. The request
// helpers are defined in RemoteDatabase.cpp, the source code helpers in RemoteDatabaseSourceCode.cpp. The request
// and response bodies are the wire types in Wire/.

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <json.hpp>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Database/Remote/RemoteDatabase.h>

namespace ktt::db
{

class RemoteClient;

/** @fn std::optional<nlohmann::json> CallRemote(const RemoteClient& client, const std::string& operation,
 * std::string_view method, const std::string& path, const nlohmann::json* body, bool notFoundIsEmpty, long* status)
 * Sends one request and returns its JSON body, mapping the statuses as every RemoteDatabase method does (see Call in
 * RemoteDatabase.cpp): 200 and 201 succeed, a 404 with a ktt-db error body is std::nullopt when notFoundIsEmpty, and
 * everything else throws a KttException that names the operation and never contains the API key.
 * @param status If not null, receives the status of a successful response (200 or 201).
 */
std::optional<nlohmann::json> CallRemote(const RemoteClient &client, const std::string &operation,
    std::string_view method, const std::string &path, const nlohmann::json *body, bool notFoundIsEmpty,
    long *status = nullptr);

/** @fn void TranslateRemote(const std::string& operation, const RemoteClient& client,
 * const std::function<void()>& function)
 * Runs the body of a public method so that a JSON error or a KttException becomes a KttException that starts with the
 * operation and has no API key (see Translate in RemoteDatabase.cpp).
 */
void TranslateRemote(const std::string &operation, const RemoteClient &client, const std::function<void()> &function);

/** @fn size_t GetSourceFingerprint(const std::vector<SourceDefinition>& definitions)
 * The source fingerprint of the definitions, computed as FingerprintUtility::GetFingerprintOfDefinitions computes it
 * from a kernel's definitions: an order-sensitive XXH3 combination of the sources. The names are not part of it.
 */
size_t GetSourceFingerprint(const std::vector<SourceDefinition> &definitions);

/** @fn void UploadMissingSourceCode(const RemoteDatabase& database, const SourceCodeProvider& provider,
 * size_t sourceFingerprint, std::optional<bool> sourceHasCode)
 * The automatic upload after a save (cpp-requirements.md R4.1.8, R6.3): when a provider is set and the POST /run
 * response says "source_has_code": false, asks the provider for the code and uploads it with SaveSourceCode. A failed
 * upload does not throw: the results are already stored, so every failure (one of the provider included) is logged as
 * a warning.
 * @param sourceHasCode The source_has_code of the response; std::nullopt when the response does not say, which is
 * logged as a warning and uploads nothing.
 */
void UploadMissingSourceCode(const RemoteDatabase &database, const SourceCodeProvider &provider,
    size_t sourceFingerprint, std::optional<bool> sourceHasCode);

} // namespace ktt::db
