#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <Api/Info/DatabaseTuningInfo.h>
#include <Database/IDatabase.h>
#include <Database/Remote/Utility/Wire/Common.h>

namespace ktt::db::wire
{

/** @struct SourceStatsResponse
 * The response of GET /source/{fingerprint}/stats (SourceStatsResponse in ktt_db.json).
 */
struct SourceStatsResponse
{
    size_t spaceCount{};
    size_t deviceCount{};
    size_t runCount{};
    size_t resultCount{};

    /** @fn SourceStats ToSourceStats() const
     * The statistics without a name: the application API returns none (cpp-requirements.md R1.4).
     */
    SourceStats ToSourceStats() const;
};

/** @struct SourceDefinitionIn
 * One kernel definition of a source (SourceDefinitionIn in ktt_db.json).
 */
struct SourceDefinitionIn
{
    std::string name;
    std::string source;
};

/** @struct SaveSourceCodeRequest
 * The body of PUT /source/{fingerprint}/code (SaveSourceCodeRequest in ktt_db.json).
 */
struct SaveSourceCodeRequest
{
    std::vector<SourceDefinitionIn> definitions;

    static SaveSourceCodeRequest From(const std::vector<SourceDefinition> &definitions);
};

void from_json(const json &source, SourceStatsResponse &response);
void to_json(json &target, const SourceDefinitionIn &definition);
void to_json(json &target, const SaveSourceCodeRequest &request);

} // namespace ktt::db::wire
