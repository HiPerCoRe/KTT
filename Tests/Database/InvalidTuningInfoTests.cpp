#include <catch.hpp>

#if defined(KTT_DATABASE)

#include "DatabaseTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database rejects saving with an invalid TuningInfo", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));

    auto tuningInfo = MakeTuningInfo();
    tuningInfo.spaceInfo.sourceFingerprint = 0; // fingerprints must all be non-zero

    REQUIRE_THROWS_AS(db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)}), ktt::KttException);
}

TEST_CASE("Database rejects saving without a tuner version", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));

    auto tuningInfo = MakeTuningInfo();
    tuningInfo.tuner.version = "";

    REQUIRE_THROWS_AS(db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)}), ktt::KttException);
}

#endif // KTT_DATABASE
