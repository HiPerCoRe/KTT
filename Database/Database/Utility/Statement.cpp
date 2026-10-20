#include <algorithm>
#include <cctype>

#include <Api/KttException.h>
#include <Database/Utility/Statement.h>

namespace ktt::db
{

// Collapses all whitespace runs (including newlines of multi-line raw strings) into single spaces.
static std::string CollapseWhitespace(const char* text)
{
    std::string result;
    bool pendingSpace = false;

    for (const char* c = text; *c != '\0'; ++c)
    {
        if (std::isspace(static_cast<unsigned char>(*c)))
        {
            pendingSpace = true;
            continue;
        }

        if (pendingSpace && !result.empty())
            result += ' ';

        pendingSpace = false;
        result += *c;
    }

    return result;
}

Statement::Statement(sqlite3* connection, const char* sql) :
    m_Connection(connection),
    m_Statement(nullptr),
    m_Sql(CollapseWhitespace(sql))
{
    if (sqlite3_prepare_v2(m_Connection, sql, -1, &m_Statement, nullptr) != SQLITE_OK)
    {
        // sqlite3_prepare_v2 leaves the handle null on failure, so there is nothing to finalize.
        ThrowError("prepare", sqlite3_errmsg(m_Connection));
    }
}

Statement::Statement(sqlite3* connection, const std::string& sql) :
    Statement(connection, sql.c_str())
{}

Statement::~Statement()
{
    sqlite3_finalize(m_Statement);
}

void Statement::BindInt(const int index, const int value)
{
    sqlite3_bind_int(m_Statement, index, value);
}

void Statement::BindInt64(const int index, const int64_t value)
{
    sqlite3_bind_int64(m_Statement, index, static_cast<sqlite3_int64>(value));
}

void Statement::BindText(const int index, const std::string& value)
{
    sqlite3_bind_text(m_Statement, index, value.c_str(), -1, SQLITE_TRANSIENT);
}

void Statement::BindNull(const int index)
{
    sqlite3_bind_null(m_Statement, index);
}

void Statement::BindUuid(const int index, const uuid& value)
{
    sqlite3_bind_blob(m_Statement, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
}

void Statement::BindOptionalInt(const int index, const std::optional<int>& value)
{
    if (value)
        BindInt(index, *value);
    else
        BindNull(index);
}

void Statement::BindOptionalText(const int index, const std::optional<std::string>& value)
{
    if (value)
        BindText(index, *value);
    else
        BindNull(index);
}

bool Statement::Step()
{
    const int result = sqlite3_step(m_Statement);

    if (result == SQLITE_ROW)
        return true;

    if (result == SQLITE_DONE)
        return false;

    ThrowError("execute", sqlite3_errmsg(m_Connection));
}

void Statement::Execute()
{
    if (Step())
        ThrowError("execute", "unexpected result row");
}

void Statement::Reset()
{
    sqlite3_reset(m_Statement);
    sqlite3_clear_bindings(m_Statement);
}

bool Statement::IsNull(const int column) const
{
    return sqlite3_column_type(m_Statement, column) == SQLITE_NULL;
}

int Statement::GetInt(const int column) const
{
    return sqlite3_column_int(m_Statement, column);
}

int64_t Statement::GetInt64(const int column) const
{
    return static_cast<int64_t>(sqlite3_column_int64(m_Statement, column));
}

size_t Statement::GetSizeT(const int column) const
{
    return static_cast<size_t>(sqlite3_column_int64(m_Statement, column));
}

std::string Statement::GetText(const int column) const
{
    const auto* text = reinterpret_cast<const char*>(sqlite3_column_text(m_Statement, column));
    return text != nullptr ? text : "";
}

std::optional<std::string> Statement::GetOptionalText(const int column) const
{
    if (IsNull(column))
        return std::nullopt;

    return GetText(column);
}

uuid Statement::GetUuid(const int column) const
{
    const auto* bytes = static_cast<const std::uint8_t*>(sqlite3_column_blob(m_Statement, column));
    const int size = sqlite3_column_bytes(m_Statement, column);

    if (bytes == nullptr || size != static_cast<int>(uuid{}.size()))
        ThrowError("read UUID column of", "expected a 16-byte BLOB");

    uuid id;
    std::copy(bytes, bytes + size, id.begin());
    return id;
}

void Statement::ThrowError(const std::string& action, const std::string& reason) const
{
    throw KttException("Failed to " + action + " statement \"" + m_Sql + "\": " + reason);
}

} // namespace ktt::db
