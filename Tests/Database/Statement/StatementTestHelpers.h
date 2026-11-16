#pragma once

// Shared fixture for the Statement unit tests. Only included from within the #if defined(KTT_DATABASE) guard of each
// test file, like DatabaseTestHelpers.h.

#include <sqlite3.h>

namespace ktt::db::test
{

// Private in-memory SQLite connection with a small `item` table (unique names) to run statements against.
struct StatementTestConnection
{
    sqlite3* handle = nullptr;

    StatementTestConnection()
    {
        sqlite3_open(":memory:", &handle);
        sqlite3_exec(handle, "CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)", nullptr, nullptr,
            nullptr);
    }

    ~StatementTestConnection()
    {
        sqlite3_close(handle);
    }

    StatementTestConnection(const StatementTestConnection&) = delete;
    StatementTestConnection& operator=(const StatementTestConnection&) = delete;

    // Number of prepared statements that have not been finalized.
    int OpenStatements() const
    {
        int count = 0;
        for (sqlite3_stmt* statement = sqlite3_next_stmt(handle, nullptr); statement != nullptr;
             statement = sqlite3_next_stmt(handle, statement))
        {
            ++count;
        }
        return count;
    }
};

} // namespace ktt::db::test
