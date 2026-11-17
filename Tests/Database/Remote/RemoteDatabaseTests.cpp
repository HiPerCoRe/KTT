#include <catch.hpp>

#if defined(KTT_REMOTE_DATABASE)

#include <memory>
#include <string>
#include <type_traits>

#include <Api/KttException.h>
#include <Database/Remote/RemoteDatabase.h>

#if defined(KTT_DATABASE)
#include <Database/Local/Database.h>
#endif

#include "FakeHttpClient.h"

using ktt::db::RemoteDatabase;
using ktt::db::RemoteDatabaseOptions;

static_assert(std::is_base_of_v<ktt::db::IDatabase, RemoteDatabase>);
static_assert(!std::is_copy_constructible_v<RemoteDatabase>);
static_assert(!std::is_copy_assignable_v<RemoteDatabase>);

namespace
{

// A function that takes "a database" accepts both implementations (R2).
bool AcceptsDatabase(const ktt::db::IDatabase &)
{
    return true;
}

} // namespace

TEST_CASE("RemoteDatabaseOptions have the documented defaults", "[Database][Remote]")
{
    const RemoteDatabaseOptions options;

    REQUIRE(options.connectTimeout == std::chrono::milliseconds(5000));
    REQUIRE(options.requestTimeout == std::chrono::milliseconds(60000));
    REQUIRE(options.maxRetries == 3);
    REQUIRE(options.verifyTls);
    REQUIRE_FALSE(options.caBundle.has_value());
}

TEST_CASE("RemoteDatabase construction does not contact the server", "[Database][Remote]")
{
    const auto http = std::make_shared<ktt::db::test::FakeHttpClient>();
    const RemoteDatabase db("kttak-test", {}, http);

    REQUIRE(http->Requests().empty());
    REQUIRE(AcceptsDatabase(db));
#if defined(KTT_DATABASE)
    REQUIRE(AcceptsDatabase(ktt::db::Database(std::filesystem::path(":memory:"))));
#endif
}

TEST_CASE("RemoteDatabase with libcurl is constructed without a request", "[Database][Remote]")
{
    // The constructor must succeed whether or not a server runs at the host, because it sends nothing.
    REQUIRE_NOTHROW(RemoteDatabase("kttak-test"));
}

TEST_CASE("RemoteDatabase talks to the host fixed at build time", "[Database][Remote]")
{
    const std::string host = RemoteDatabase::GetHost();

    REQUIRE((host.rfind("http://", 0) == 0 || host.rfind("https://", 0) == 0));
    REQUIRE(host.back() != '/');

    const auto http = std::make_shared<ktt::db::test::FakeHttpClient>();
    http->EnqueueResponse(404, R"({"error":"source not found"})");
    const RemoteDatabase db("kttak-test", {}, http);
    static_cast<void>(db.GetStatsForSource(42));

    const auto requests = http->Requests();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests[0].url.rfind(host + "/ktt-db/app/", 0) == 0);
}

TEST_CASE("RemoteDatabase rejects invalid arguments", "[Database][Remote]")
{
    const auto http = std::make_shared<ktt::db::test::FakeHttpClient>();

    REQUIRE_THROWS_AS(RemoteDatabase("", {}, http), ktt::KttException);
    REQUIRE_THROWS_AS(RemoteDatabase("kttak-test", {}, nullptr), ktt::KttException);
    REQUIRE_THROWS_AS(RemoteDatabase(""), ktt::KttException);
}

#endif // KTT_REMOTE_DATABASE
