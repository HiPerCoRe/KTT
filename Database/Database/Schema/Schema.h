#pragma once

#include <sqlite3.h>
namespace ktt::db
{

/** @class Schema
 * Manages the SQLite database schema creation and initialization.
 * Provides static methods to ensure all required database tables, indexes,
 * and constraints are created.
 */
class Schema
{
public:
    Schema() = delete;

    /** @fn static void CreateIfNotExists(sqlite3* connection)
     * Creates the database schema if it does not already exist.
     * Creates tables for compute API, device info, device API, tuning source, tuning space,
     * tuning run, and tuning result. Also creates unique indexes and foreign key constraints.
     * @param connection Pointer to the SQLite connection to execute schema creation.
     * @throw std::runtime_error If schema creation fails.
     */
    static void CreateIfNotExists(sqlite3* connection);
};

} // namespace ktt::db
