#include <sqlite3.h>


namespace ktt::db
{

/** @class TransactionGuard
 * RAII helper that opens a SQLite transaction on construction and rolls it back
 * on destruction unless Commit() has been called. Guarantees the transaction is
 * not left open if an exception propagates out of the enclosing scope.
 */
class TransactionGuard
{
public:
    /** @fn explicit TransactionGuard(sqlite3* connection)
     * Begins a transaction on the given connection.
     * @param connection SQLite database connection.
     * @throw KttException If the transaction cannot be started.
     */
    explicit TransactionGuard(sqlite3* connection);

    /** Rolls the transaction back if it has not been committed. */
    ~TransactionGuard();

    TransactionGuard(const TransactionGuard&) = delete;
    TransactionGuard& operator=(const TransactionGuard&) = delete;

    /** @fn void Commit()
     * Commits the transaction. After a successful commit the guard no longer
     * rolls back on destruction.
     * @throw KttException If the COMMIT statement fails.
     */
    void Commit();

private:
    sqlite3* Connection;
    bool Committed = false;
};

} // namespace ktt::db
