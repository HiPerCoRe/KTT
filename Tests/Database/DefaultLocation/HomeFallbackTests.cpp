#include <catch.hpp>

#if defined(KTT_DATABASE) && !defined(_MSC_VER)

#include <filesystem>

#include <Database/Database.h>

#include "DefaultLocationTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database falls back to HOME/.local/share when XDG_DATA_HOME is not set", "Database")
{
    const auto home = std::filesystem::temp_directory_path() / "ktt_database_test_home";
    std::filesystem::remove_all(home);

    {
        const ScopedEnvironmentVariable xdgDataHome("XDG_DATA_HOME", std::nullopt);
        const ScopedEnvironmentVariable homeVariable("HOME", home.string());
        const ktt::db::Database db;
    }

    REQUIRE(std::filesystem::exists(home / ".local" / "share" / "ktt" / "ktt.db"));

    std::filesystem::remove_all(home);
}

#endif // KTT_DATABASE && !_MSC_VER
