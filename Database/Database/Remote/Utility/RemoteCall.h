#pragma once

// Internal to the remote database client (not exported): sending a request and mapping its response and errors as
// every RemoteDatabase method does.

#include <exception>
#include <optional>
#include <string>
#include <string_view>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/Utility/RemoteClient.h>

namespace ktt::db
{

/** @fn std::optional<nlohmann::json> CallRemote(const RemoteClient& client, const std::string& operation,
 * std::string_view method, const std::string& path, const nlohmann::json* body, bool notFoundIsEmpty, long* status)
 * Sends one request and returns its JSON body. Every request may be retried (RetryPolicy::Idempotent): the reads are
 * safe to repeat, and POST /run carries the run id, so the server recognizes a repeated save.
 * @param status If not null, receives the status of a successful response (200 or 201).
 * @return The body of a 200 or 201 response; std::nullopt for 404 when notFoundIsEmpty and the body is a ktt-db error
 *         body. Any other 404 (an unmatched route has an empty body, a proxy sends its own page) throws, so a wrong base
 *         URL is not mistaken for "not found".
 * @throw KttException For every other status, a network error, or a success body that is not a JSON object. The
 *        message starts with the operation and has the request and the status, never the API key.
 */
std::optional<nlohmann::json> CallRemote(const RemoteClient &client, const std::string &operation,
    std::string_view method, const std::string &path, const nlohmann::json *body, bool notFoundIsEmpty,
    long *status = nullptr);

/** @struct PredicateError
 * Carries an exception thrown by a user's predicate to TranslateRemote, which rethrows the original exception
 * unchanged.
 */
struct PredicateError
{
    std::exception_ptr error;
};

/** @fn bool CallPredicate(const Predicate& predicate, const Argument& argument)
 * Calls a user's predicate inside TranslateRemote, so that its exceptions (a KttException included) reach the caller
 * unchanged instead of being prefixed with the operation.
 */
template <typename Predicate, typename Argument>
bool CallPredicate(const Predicate &predicate, const Argument &argument)
{
    try
    {
        return predicate(argument);
    }
    catch (...)
    {
        throw PredicateError{std::current_exception()};
    }
}

/** @fn auto TranslateRemote(const std::string& operation, const RemoteClient& client, Function function)
 * Runs the body of a public method so that every error names the method: a JSON error (a missing field or a wrong
 * type in a response, or request text that is not valid UTF-8) and a KttException from the conversions become a
 * KttException that starts with the operation and has no API key. Other exceptions pass unchanged; a user's predicate
 * is called through CallPredicate, so its exceptions reach the caller unchanged.
 * @return What function returns.
 */
template <typename Function>
auto TranslateRemote(const std::string &operation, const RemoteClient &client, Function function)
    -> decltype(function())
{
    try
    {
        return function();
    }
    catch (const PredicateError &error)
    {
        std::rethrow_exception(error.error);
    }
    catch (const nlohmann::json::exception &error)
    {
        throw KttException(operation + " failed: invalid JSON: " + client.Redact(error.what()));
    }
    catch (const KttException &error)
    {
        const std::string message = error.what();

        if (message.rfind(operation, 0) == 0)
            throw;

        throw KttException(operation + " failed: " + client.Redact(message), error.GetReason());
    }
}

} // namespace ktt::db
