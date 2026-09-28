#include <catch.hpp>

#if defined(KTT_DATABASE)

#include "DatabaseTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database returns nothing for a source that was never saved", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));

    const auto loaded = db.SimpleGetBestResults(MakeTuningInfo());

    REQUIRE(loaded.empty());
}

#endif // KTT_DATABASE
