#pragma once

#include <cstdint>
#include <optional>
#include <sqlite3.h>
#include <string>

#include <Database/Utility/Uuid.h>

namespace ktt::db
{

/** @class Statement
 * RAII wrapper around a prepared SQLite statement. The statement is finalized on destruction, including when an
 * exception propagates out of the enclosing scope. Errors are reported as KttException naming the failing SQL
 * (collapsed to a single line), e.g. `Failed to execute statement "INSERT INTO tuner ...": UNIQUE constraint failed`.
 * Bind indices are 1-based and column indices are 0-based, as in the SQLite C API.
 */
class Statement
{
public:
    /** @fn Statement(sqlite3* connection, const char* sql)
     * Prepares the statement.
     * @param connection SQLite database connection.
     * @param sql SQL text of the statement.
     * @throw KttException If the statement cannot be prepared.
     */
    Statement(sqlite3* connection, const char* sql);

    /** @fn Statement(sqlite3* connection, const std::string& sql)
     * Prepares the statement. See the const char* overload.
     */
    Statement(sqlite3* connection, const std::string& sql);

    /** Finalizes the statement. */
    ~Statement();

    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    void BindInt(const int index, const int value);
    void BindInt64(const int index, const int64_t value);
    void BindText(const int index, const std::string& value);
    void BindNull(const int index);

    /** @fn void BindUuid(const int index, const uuid& value)
     * Binds a UUID as a 16-byte BLOB.
     */
    void BindUuid(const int index, const uuid& value);

    /** @fn void BindOptionalInt(const int index, const std::optional<int>& value)
     * Binds the value, or NULL when it is unset.
     */
    void BindOptionalInt(const int index, const std::optional<int>& value);

    /** @fn void BindOptionalText(const int index, const std::optional<std::string>& value)
     * Binds the value, or NULL when it is unset.
     */
    void BindOptionalText(const int index, const std::optional<std::string>& value);

    /** @fn bool Step()
     * Advances to the next result row.
     * @return True if a row is available, false when the statement has finished.
     * @throw KttException If the step fails.
     */
    bool Step();

    /** @fn void Execute()
     * Runs a statement that returns no rows (INSERT, UPDATE, ...).
     * @throw KttException If the statement does not complete successfully.
     */
    void Execute();

    /** @fn void Reset()
     * Resets the statement and clears its bindings so it can be executed again with new values.
     */
    void Reset();

    bool IsNull(const int column) const;
    int GetInt(const int column) const;
    int64_t GetInt64(const int column) const;
    size_t GetSizeT(const int column) const;

    /** @fn std::string GetText(const int column) const
     * Returns the column as text, or an empty string when it is NULL.
     */
    std::string GetText(const int column) const;

    /** @fn std::optional<std::string> GetOptionalText(const int column) const
     * Returns the column as text, or std::nullopt when it is NULL.
     */
    std::optional<std::string> GetOptionalText(const int column) const;

    /** @fn uuid GetUuid(const int column) const
     * Reads a UUID from a 16-byte BLOB column.
     * @throw KttException If the column is not a 16-byte BLOB.
     */
    uuid GetUuid(const int column) const;

private:
    sqlite3* m_Connection;
    sqlite3_stmt* m_Statement;
    std::string m_Sql; ///< SQL text collapsed to a single line, for error messages.

    [[noreturn]] void ThrowError(const std::string& action, const std::string& reason) const;
};

} // namespace ktt::db
