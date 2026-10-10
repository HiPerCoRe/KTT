#include <catch.hpp>

#if defined(KTT_DATABASE)

#include <stdexcept>
#include <string>

#include <Database/Utility/Statement.h>

#include "StatementTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Statement is finalized when an exception leaves its scope", "Database")
{
    StatementTestConnection connection;
    sqlite3_exec(connection.handle, "INSERT INTO item (name) VALUES ('not a number')", nullptr, nullptr, nullptr);

    // Mirrors the repositories' read loops, where parsing a column (std::stoull) can throw mid-iteration.
    const auto readAll = [&connection]()
    {
        ktt::db::Statement select(connection.handle, "SELECT name FROM item");
        while (select.Step())
            static_cast<void>(std::stoull(select.GetText(0)));
    };

    REQUIRE_THROWS_AS(readAll(), std::invalid_argument);
    REQUIRE(connection.OpenStatements() == 0);
}

#endif // KTT_DATABASE
