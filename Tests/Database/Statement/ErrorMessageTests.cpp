#include <catch.hpp>

#if defined(KTT_DATABASE)

#include <string>

#include <Api/KttException.h>
#include <Database/Utility/Statement.h>

#include "StatementTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Statement error messages name the failing SQL", "Database")
{
    StatementTestConnection connection;

    SECTION("Prepare error")
    {
        try
        {
            ktt::db::Statement statement(connection.handle, R"(
                SELECT id
                FROM missing_table
            )");
            FAIL("Expected KttException");
        }
        catch (const ktt::KttException& exception)
        {
            const std::string message = exception.what();
            REQUIRE(message.find("Failed to prepare statement \"SELECT id FROM missing_table\"") != std::string::npos);
            REQUIRE(message.find("no such table: missing_table") != std::string::npos);
        }
    }

    SECTION("Execute error")
    {
        ktt::db::Statement insert(connection.handle, "INSERT INTO item (name) VALUES (?)");
        insert.BindText(1, "duplicate");
        insert.Execute();
        insert.Reset();
        insert.BindText(1, "duplicate");

        try
        {
            insert.Execute();
            FAIL("Expected KttException");
        }
        catch (const ktt::KttException& exception)
        {
            const std::string message = exception.what();
            REQUIRE(message.find("Failed to execute statement \"INSERT INTO item (name) VALUES (?)\"") != std::string::npos);
            REQUIRE(message.find("UNIQUE constraint failed") != std::string::npos);
        }
    }
}

#endif // KTT_DATABASE
