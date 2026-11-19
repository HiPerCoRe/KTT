#pragma once

// Internal to the remote database client (not exported): the source code helpers that RemoteDatabase.cpp shares with
// RemoteDatabaseSourceCode.cpp, where they are defined. The request helpers are in RemoteCall.h, the request and
// response bodies are the wire types in Wire/.

#include <cstddef>
#include <optional>
#include <vector>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Database/Remote/RemoteDatabase.h>

namespace ktt::db
{

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
