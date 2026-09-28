#include <catch.hpp>

#if defined(KTT_DATABASE)

#include "DatabaseTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database stores a result and loads it back", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    const auto tuningInfo = MakeTuningInfo();

    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)});

    const auto loaded = db.SimpleGetBestResults(tuningInfo);

    REQUIRE(loaded.size() == 1);
    REQUIRE(loaded[0].GetKernelName() == "simpleKernel");
    REQUIRE(loaded[0].GetStatus() == ktt::ResultStatus::Ok);
    REQUIRE(loaded[0].GetKernelDuration() == 5 * Millisecond);

    const auto& pairs = loaded[0].GetConfiguration().GetPairs();
    REQUIRE(pairs.size() == 1);
    REQUIRE(pairs[0].GetName() == "block_size");
    REQUIRE(pairs[0].GetValueUint() == 64);
}

#endif // KTT_DATABASE
