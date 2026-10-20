#include <catch.hpp>

#if defined(KTT_DATABASE)

#include <Database/Utility/Statement.h>

#include "StatementTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Statement inserts and reads rows", "Database")
{
    StatementTestConnection connection;

    {
        ktt::db::Statement insert(connection.handle, "INSERT INTO item (name) VALUES (?)");
        insert.BindText(1, "first");
        insert.Execute();
        insert.Reset();
        insert.BindText(1, "second");
        insert.Execute();
    }

    ktt::db::Statement select(connection.handle, "SELECT id, name FROM item ORDER BY id");
    REQUIRE(select.Step());
    REQUIRE(select.GetSizeT(0) == 1);
    REQUIRE(select.GetText(1) == "first");
    REQUIRE(select.Step());
    REQUIRE(select.GetText(1) == "second");
    REQUIRE_FALSE(select.Step());
}

#endif // KTT_DATABASE
