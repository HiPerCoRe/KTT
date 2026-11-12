#include <catch.hpp>

#if defined(KTT_DATABASE) && !defined(_MSC_VER)

#include <Api/KttException.h>
#include <Database/Database.h>

#include "DefaultLocationTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database reports an error when no default location can be determined", "Database")
{
    const ScopedEnvironmentVariable xdgDataHome("XDG_DATA_HOME", std::nullopt);
    const ScopedEnvironmentVariable home("HOME", std::nullopt);

    REQUIRE_THROWS_AS(ktt::db::Database(), ktt::KttException);
}

#endif // KTT_DATABASE && !_MSC_VER
