#include <string>

#include <Api/KttException.h>
#include <Database/Utility/TransactionGuard.h>

namespace ktt::db
{

TransactionGuard::TransactionGuard(sqlite3* connection) : Connection(connection)
{
    char* errorMessage = nullptr;
    if (sqlite3_exec(Connection, "BEGIN", nullptr, nullptr, &errorMessage) != SQLITE_OK)
    {
        const std::string error = errorMessage != nullptr ? errorMessage : "unknown error";
        sqlite3_free(errorMessage);
        throw KttException("Failed to begin transaction: " + error);
    }
}

TransactionGuard::~TransactionGuard()
{
    if (!Committed)
        sqlite3_exec(Connection, "ROLLBACK", nullptr, nullptr, nullptr);
}

void TransactionGuard::Commit()
{
    char* errorMessage = nullptr;
    if (sqlite3_exec(Connection, "COMMIT", nullptr, nullptr, &errorMessage) != SQLITE_OK)
    {
        const std::string error = errorMessage != nullptr ? errorMessage : "unknown error";
        sqlite3_free(errorMessage);
        throw KttException("Failed to commit transaction: " + error);
    }
    Committed = true;
}

} // namespace ktt::db
