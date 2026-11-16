#include <catch.hpp>

#if defined(KTT_DATABASE)

#include "DatabaseTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database persists results across reopening the file", "Database")
{
    const auto path = std::filesystem::temp_directory_path() / "ktt_database_test_persistence.db";
    std::filesystem::remove(path);

    const auto tuningInfo = MakeTuningInfo();

    {
        const ktt::db::Database db(path);
        db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 7 * Millisecond)});
    }

    {
        const ktt::db::Database db(path);
        const auto loaded = db.SimpleGetBestResults(tuningInfo);

        REQUIRE(loaded.size() == 1);
        REQUIRE(loaded[0].GetKernelDuration() == 7 * Millisecond);
    }

    std::filesystem::remove(path);
}

#endif // KTT_DATABASE
