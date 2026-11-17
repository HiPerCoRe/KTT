#include <catch.hpp>

#if defined(KTT_DATABASE)

#include <filesystem>
#include <type_traits>

#include <Database/IDatabase.h>
#include <Database/Local/Database.h>

#include "../DatabaseTestHelpers.h"

using namespace ktt::db::test;

namespace
{

// Takes "a database" without knowing whether it is local or remote (R2).
std::vector<ktt::KernelResult> SaveAndLoadThroughInterface(const ktt::db::IDatabase &db, const ktt::db::TuningInfo &info)
{
    db.SaveResults(info, {MakeResult("kernel", 64, 3 * Millisecond), MakeResult("kernel", 128, 1 * Millisecond)});
    return db.SimpleGetBestResults(info);
}

} // namespace

static_assert(std::is_abstract_v<ktt::db::IDatabase>);
static_assert(std::is_base_of_v<ktt::db::IDatabase, ktt::db::Database>);
static_assert(std::has_virtual_destructor_v<ktt::db::IDatabase>);

TEST_CASE("Database is usable through the IDatabase interface", "[Database][IDatabase]")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    const auto info = MakeTuningInfo();

    const auto results = SaveAndLoadThroughInterface(db, info);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0].GetKernelDuration() == 1 * Millisecond);
    REQUIRE(results[1].GetKernelDuration() == 3 * Millisecond);

    const ktt::db::IDatabase &asInterface = db;
    const ktt::db::GetBestResultsQuery query{info.spaceInfo, std::nullopt, std::nullopt, 1};
    REQUIRE(asInterface.GetBestResults(query).size() == 1);

    const auto stats = asInterface.GetStatsForSource(info.spaceInfo.sourceFingerprint);
    REQUIRE(stats.has_value());
    REQUIRE(stats->runCount == 1);
    REQUIRE(stats->resultCount == 2);
    REQUIRE_FALSE(asInterface.GetStatsForSource(info.spaceInfo.sourceFingerprint + 1).has_value());
}

#endif // KTT_DATABASE
