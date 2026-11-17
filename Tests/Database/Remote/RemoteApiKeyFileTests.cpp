#include <catch.hpp>

#if defined(KTT_REMOTE_DATABASE) && !defined(_MSC_VER)

// The API key file next to the default local database ($XDG_DATA_HOME/ktt/api-key, ~/.local/share/ktt/api-key). The
// tests change environment variables through the POSIX setenv/unsetenv API, so they do not run with MSVC.

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>

#include <Api/KttException.h>
#include <Database/Remote/RemoteDatabase.h>

#include "../Local/DefaultLocation/DefaultLocationTestHelpers.h"
#include "FakeHttpClient.h"

using ktt::db::RemoteDatabase;
using ktt::db::test::FakeHttpClient;
using ktt::db::test::ScopedEnvironmentVariable;

namespace
{

const std::string FileKey = "kttak-key-from-the-file";

std::string ExceptionText(const std::function<void()> &action)
{
    try
    {
        action();
    }
    catch (const ktt::KttException &exception)
    {
        return exception.what();
    }

    return {};
}

// A temporary XDG_DATA_HOME, removed again at the end of the test.
struct DataHome
{
    std::filesystem::path root = std::filesystem::temp_directory_path() / "ktt_remote_api_key_test";
    ScopedEnvironmentVariable xdgDataHome{"XDG_DATA_HOME", root.string()};

    DataHome()
    {
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root / "ktt");
    }

    ~DataHome()
    {
        std::filesystem::remove_all(root);
    }

    std::filesystem::path KeyFile() const
    {
        return root / "ktt" / "api-key";
    }

    void WriteKeyFile(const std::string &content) const
    {
        std::ofstream(KeyFile(), std::ios::binary) << content;
    }
};

// Sends one request and returns the x-api-key it carried.
std::string SentApiKey(const RemoteDatabase &db, FakeHttpClient &http)
{
    http.EnqueueResponse(404, R"({"error":"source not found"})");
    static_cast<void>(db.GetStatsForSource(42));
    return http.Requests().back().Header("x-api-key").value_or("");
}

} // namespace

TEST_CASE("The API key file is next to the default local database", "[Database][Remote]")
{
    {
        const ScopedEnvironmentVariable xdgDataHome("XDG_DATA_HOME", "/data/home");
        REQUIRE(RemoteDatabase::GetDefaultApiKeyPath() == std::filesystem::path("/data/home/ktt/api-key"));
    }

    {
        const ScopedEnvironmentVariable xdgDataHome("XDG_DATA_HOME", std::nullopt);
        const ScopedEnvironmentVariable home("HOME", "/home/user");
        REQUIRE(RemoteDatabase::GetDefaultApiKeyPath() == std::filesystem::path("/home/user/.local/share/ktt/api-key"));
    }

    {
        const ScopedEnvironmentVariable xdgDataHome("XDG_DATA_HOME", std::nullopt);
        const ScopedEnvironmentVariable home("HOME", std::nullopt);
        REQUIRE_THROWS_AS(RemoteDatabase::GetDefaultApiKeyPath(), ktt::KttException);
        REQUIRE_THROWS_AS(RemoteDatabase(), ktt::KttException);
    }
}

TEST_CASE("RemoteDatabase reads the API key from the key file", "[Database][Remote]")
{
    const DataHome dataHome;
    const auto http = std::make_shared<FakeHttpClient>();

    SECTION("without surrounding whitespace")
    {
        dataHome.WriteKeyFile("  " + FileKey + "\r\n\n");
        const RemoteDatabase db({}, http);
        REQUIRE(SentApiKey(db, *http) == FileKey);
    }

    SECTION("with the libcurl client, without a request")
    {
        dataHome.WriteKeyFile(FileKey);
        REQUIRE_NOTHROW(RemoteDatabase());
    }

    SECTION("not when the application gives the key")
    {
        dataHome.WriteKeyFile(FileKey);
        const RemoteDatabase db("kttak-from-the-application", {}, http);
        REQUIRE(SentApiKey(db, *http) == "kttak-from-the-application");
    }
}

TEST_CASE("RemoteDatabase does not need a key file when the application gives the key", "[Database][Remote]")
{
    const ScopedEnvironmentVariable xdgDataHome("XDG_DATA_HOME", std::nullopt);
    const ScopedEnvironmentVariable home("HOME", std::nullopt);
    const auto http = std::make_shared<FakeHttpClient>();

    REQUIRE_NOTHROW(RemoteDatabase("kttak-test", {}, http));
}

TEST_CASE("RemoteDatabase reports a missing or invalid key file without its content", "[Database][Remote]")
{
    const DataHome dataHome;
    const auto http = std::make_shared<FakeHttpClient>();
    const std::string path = dataHome.KeyFile().string();

    SECTION("missing")
    {
        const std::string text = ExceptionText([&] { RemoteDatabase db({}, http); });
        REQUIRE(text.find(path) != std::string::npos);
        REQUIRE(text.find("does not exist") != std::string::npos);
    }

    SECTION("a directory")
    {
        std::filesystem::create_directories(dataHome.KeyFile());
        REQUIRE(ExceptionText([&] { RemoteDatabase db({}, http); }).find("is not a file") != std::string::npos);
    }

    SECTION("empty or only whitespace")
    {
        for (const std::string content : {"", " \n\t\r\n"})
        {
            dataHome.WriteKeyFile(content);
            const std::string text = ExceptionText([&] { RemoteDatabase db({}, http); });
            REQUIRE(text.find(path) != std::string::npos);
            REQUIRE(text.find("is empty") != std::string::npos);
        }
    }

    SECTION("too large")
    {
        dataHome.WriteKeyFile(std::string(5000, 'k'));
        REQUIRE(ExceptionText([&] { RemoteDatabase db({}, http); }).find("too large") != std::string::npos);
    }

    SECTION("a key with a line break inside")
    {
        dataHome.WriteKeyFile(FileKey + "\n" + FileKey);
        const std::string text = ExceptionText([&] { RemoteDatabase db({}, http); });
        REQUIRE_FALSE(text.empty());
        REQUIRE(text.find(FileKey) == std::string::npos);
    }

    REQUIRE(http->Requests().empty());
}

#endif // KTT_REMOTE_DATABASE && !_MSC_VER
