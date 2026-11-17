#include <catch.hpp>

#if defined(KTT_DATABASE)

#include <filesystem>
#include <sqlite3.h>
#include <string>

#include <Database/Local/Database.h>
#include <Database/Utility/TimeUnitGuard.h>
#include <Output/TimeConfiguration/TimeConfiguration.h>

#include "../DatabaseTestHelpers.h"

using namespace ktt::db::test;

namespace
{

// Opens a private in-memory SQLite connection. The Database adopts it without owning it, so the test can inspect
// the stored rows directly through the same connection.
struct InMemoryConnection
{
    sqlite3 *handle = nullptr;

    InMemoryConnection()
    {
        REQUIRE(sqlite3_open(":memory:", &handle) == SQLITE_OK);
    }

    ~InMemoryConnection()
    {
        sqlite3_close(handle);
    }
};

// Runs a query returning a single text value.
std::string QueryText(sqlite3 *connection, const std::string &sql)
{
    sqlite3_stmt *stmt = nullptr;
    sqlite3_prepare_v2(connection, sql.c_str(), -1, &stmt, nullptr);
    REQUIRE(sqlite3_step(stmt) == SQLITE_ROW);
    const std::string value = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return value;
}

ktt::TimeUnit CurrentTimeUnit()
{
    return ktt::TimeConfiguration::GetInstance().GetTimeUnit();
}

// Two saved results of 3 ms and 5 ms, fastest first, with unscaled durations. Whole milliseconds are exact in every
// time unit up to milliseconds. The results have no extra duration, because JSON_T4 keeps only the total duration
// (as the kernel duration), so the kernel and total durations agree in all formats.
void RequireOriginalDurations(const std::vector<ktt::KernelResult> &loaded)
{
    REQUIRE(loaded.size() == 2);
    REQUIRE(loaded[0].GetKernelDuration() == 3 * Millisecond);
    REQUIRE(loaded[0].GetTotalDuration() == 3 * Millisecond);
    REQUIRE(loaded[1].GetKernelDuration() == 5 * Millisecond);
    REQUIRE(loaded[1].GetTotalDuration() == 5 * Millisecond);
}

} // namespace

TEST_CASE("Time unit names round-trip and unknown names are rejected", "Database")
{
    for (const auto unit : {ktt::TimeUnit::Nanoseconds, ktt::TimeUnit::Microseconds, ktt::TimeUnit::Milliseconds,
             ktt::TimeUnit::Seconds})
    {
        REQUIRE(ktt::db::TimeUnitFromName(ktt::db::TimeUnitToName(unit)) == unit);
    }

    REQUIRE(ktt::db::TimeUnitToName(ktt::TimeUnit::Milliseconds) == "milliseconds");
    REQUIRE_THROWS_AS(ktt::db::TimeUnitFromName("ms"), ktt::KttException);
}

TEST_CASE("TimeUnitGuard restores the previous time unit", "Database")
{
    const ktt::db::TimeUnitGuard outer(ktt::TimeUnit::Microseconds);

    SECTION("when it goes out of scope")
    {
        {
            const ktt::db::TimeUnitGuard inner(ktt::TimeUnit::Seconds);
            REQUIRE(CurrentTimeUnit() == ktt::TimeUnit::Seconds);
        }
        REQUIRE(CurrentTimeUnit() == ktt::TimeUnit::Microseconds);
    }

    SECTION("when an exception leaves its scope")
    {
        REQUIRE_THROWS([] {
            const ktt::db::TimeUnitGuard inner(ktt::TimeUnit::Seconds);
            throw ktt::KttException("failure while the guard is alive");
        }());
        REQUIRE(CurrentTimeUnit() == ktt::TimeUnit::Microseconds);
    }
}

TEST_CASE("Database loads durations in the time unit they were saved with", "Database")
{
    const auto format = GENERATE(ktt::OutputFormat::JSON, ktt::OutputFormat::JSON_T4, ktt::OutputFormat::XML);
    CAPTURE(static_cast<int>(format));

    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle);
    const auto tuningInfo = MakeTuningInfo();

    // Restores the time unit active before the test case when it ends, also when an assertion fails.
    const ktt::db::TimeUnitGuard testScope(ktt::TimeUnit::Microseconds);

    db.SaveResults(
        tuningInfo,
        {MakeResult("simpleKernel", 64, 3 * Millisecond), MakeResult("simpleKernel", 128, 5 * Millisecond)},
        {format, 2}
    );
    REQUIRE(QueryText(connection.handle, "SELECT time_unit FROM tuning_run") == "microseconds");

    // The application changes the time unit between saving and loading.
    ktt::TimeConfiguration::GetInstance().SetTimeUnit(ktt::TimeUnit::Milliseconds);

    SECTION("SimpleGetBestResults")
    {
        RequireOriginalDurations(db.SimpleGetBestResults(tuningInfo));
    }

    SECTION("GetBestResults")
    {
        RequireOriginalDurations(db.GetBestResults({tuningInfo.spaceInfo, std::nullopt, std::nullopt, 50}));
    }

    // Loading switches the unit only while it deserializes.
    REQUIRE(CurrentTimeUnit() == ktt::TimeUnit::Milliseconds);
}

TEST_CASE("Database stores the time unit per run", "Database")
{
    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle);
    const auto tuningInfo = MakeTuningInfo();

    const ktt::db::TimeUnitGuard testScope(ktt::TimeUnit::Nanoseconds);
    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 3 * Millisecond)});

    ktt::TimeConfiguration::GetInstance().SetTimeUnit(ktt::TimeUnit::Microseconds);
    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 128, 5 * Millisecond)});

    REQUIRE(QueryText(connection.handle, "SELECT time_unit FROM tuning_run WHERE id = 1") == "nanoseconds");
    REQUIRE(QueryText(connection.handle, "SELECT time_unit FROM tuning_run WHERE id = 2") == "microseconds");

    ktt::TimeConfiguration::GetInstance().SetTimeUnit(ktt::TimeUnit::Seconds);
    RequireOriginalDurations(db.GetBestResults({tuningInfo.spaceInfo, std::nullopt, std::nullopt, 50}));
}

TEST_CASE("Database sync copies the time unit of each run", "Database")
{
    const auto path = std::filesystem::temp_directory_path() / "ktt_database_test_sync_time_unit.db";
    std::filesystem::remove(path);

    const auto tuningInfo = MakeTuningInfo();
    const ktt::db::TimeUnitGuard testScope(ktt::TimeUnit::Microseconds);

    {
        const ktt::db::Database other(path);
        other.SaveResults(
            tuningInfo,
            {MakeResult("simpleKernel", 64, 3 * Millisecond), MakeResult("simpleKernel", 128, 5 * Millisecond)}
        );
    }

    ktt::TimeConfiguration::GetInstance().SetTimeUnit(ktt::TimeUnit::Milliseconds);

    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle);
    REQUIRE(db.SyncFrom(ktt::db::Database(path)) == 1);

    REQUIRE(QueryText(connection.handle, "SELECT time_unit FROM tuning_run") == "microseconds");
    RequireOriginalDurations(db.SimpleGetBestResults(tuningInfo));

    std::filesystem::remove(path);
}

#endif // KTT_DATABASE
