#include <catch.hpp>

#if defined(KTT_DATABASE) && !defined(_MSC_VER)

#include <filesystem>

#include <Database/Database.h>

#include "DefaultLocationTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database defaults to XDG_DATA_HOME when it is set", "Database")
{
    const auto dataHome = std::filesystem::temp_directory_path() / "ktt_database_test_xdg_data_home";
    std::filesystem::remove_all(dataHome);

    {
        const ScopedEnvironmentVariable xdgDataHome("XDG_DATA_HOME", dataHome.string());
        const ktt::db::Database db;
    }

    REQUIRE(std::filesystem::exists(dataHome / "ktt" / "ktt.db"));

    std::filesystem::remove_all(dataHome);
}

#endif // KTT_DATABASE && !_MSC_VER
