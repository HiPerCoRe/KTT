#include <catch.hpp>

#if defined(KTT_DATABASE)

#include "DatabaseTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database returns best results fastest-first and honours the limit", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    const auto tuningInfo = MakeTuningInfo();

    db.SaveResults(tuningInfo, {
        MakeResult("simpleKernel", 32, 8 * Millisecond),
        MakeResult("simpleKernel", 64, 3 * Millisecond),
        MakeResult("simpleKernel", 128, 5 * Millisecond),
    });

    SECTION("All results are returned ordered by kernel duration ascending")
    {
        const auto loaded = db.SimpleGetBestResults(tuningInfo);

        REQUIRE(loaded.size() == 3);
        REQUIRE(loaded[0].GetKernelDuration() == 3 * Millisecond);
        REQUIRE(loaded[1].GetKernelDuration() == 5 * Millisecond);
        REQUIRE(loaded[2].GetKernelDuration() == 8 * Millisecond);
    }

    SECTION("Only the fastest results are returned when a limit is given")
    {
        const auto loaded = db.SimpleGetBestResults(tuningInfo, 2);

        REQUIRE(loaded.size() == 2);
        REQUIRE(loaded[0].GetKernelDuration() == 3 * Millisecond);
        REQUIRE(loaded[1].GetKernelDuration() == 5 * Millisecond);
    }
}

#endif // KTT_DATABASE
