#include <catch.hpp>

#if defined(KTT_REMOTE_DATABASE)

// R6, R8.1, R8.3: RemoteDatabase::SaveSourceCode and the automatic upload after SaveResults, against a scripted fake
// HttpClient.

#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <json.hpp>

#include <Api/KttException.h>
#include <Database/Remote/RemoteDatabase.h>
#include <Database/Remote/Utility/RemoteInternal.h>
#include <Utility/Logger/Logger.h>

#include "../DatabaseTestHelpers.h"
#include "FakeHttpClient.h"

using json = nlohmann::json;
using ktt::db::GetSourceFingerprint;
using ktt::db::RemoteDatabase;
using ktt::db::RemoteDatabaseOptions;
using ktt::db::SourceCodeProvider;
using ktt::db::SourceDefinition;
using ktt::db::test::FakeHttpClient;
using ktt::db::test::MakeResult;
using ktt::db::test::MakeTuningInfo;
using ktt::db::test::Millisecond;
using ktt::db::test::RecordedRequest;

// MakeSourceCodeProvider needs a Tuner, which the tests cannot create (no device), so only its type is checked.
static_assert(std::is_same_v<decltype(&ktt::db::MakeSourceCodeProvider),
    SourceCodeProvider (*)(const ktt::Tuner &, std::vector<ktt::KernelId>)>);

namespace
{

const std::string ApiKey = "kttak-secret-test-key";
const std::string BaseUrl = RemoteDatabase::GetHost();
const std::string AppUrl = BaseUrl + "/ktt-db/app";
const std::string Operation = "RemoteDatabase::SaveSourceCode";

const std::string SourceA = "__kernel void a(__global float* x) { x[0] = 1.0f; }\n";
const std::string SourceB = "__global__ void b(float* x) { x[0] = 2.0f; }\n";
// The fingerprint of [SourceA, SourceB] (the "two_ab" test vector).
constexpr size_t FingerprintAB = 16551293918983522932ULL;

const std::string CodeResponse = R"({"source_id":1,"definition_count":2})";

std::string SaveRunResponse(const char *sourceHasCode)
{
    return std::string(R"({"run_id":"0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b17","source_id":1,"space_id":2,)")
        + R"("result_count":1,"realm_ids":["00000000-0000-4000-8000-000000000001"])"
        + (sourceHasCode != nullptr ? std::string(R"(,"source_has_code":)") + sourceHasCode : std::string()) + "}";
}

struct Fixture
{
    std::shared_ptr<FakeHttpClient> http = std::make_shared<FakeHttpClient>();

    RemoteDatabase MakeDatabase(SourceCodeProvider provider = {}, const uint32_t maxRetries = 0) const
    {
        RemoteDatabaseOptions options;
        options.maxRetries = maxRetries;
        options.sourceCodeProvider = std::move(provider);
        return RemoteDatabase(ApiKey, options, http);
    }

    std::vector<RecordedRequest> Requests() const
    {
        return http->Requests();
    }
};

// Captures the KTT log for the lifetime of the object; then logs to std::clog again, KTT's default target.
class LogCapture
{
public:
    LogCapture() :
        m_Level(ktt::Logger::GetLogger().GetLoggingLevel())
    {
        ktt::Logger::GetLogger().SetLoggingLevel(ktt::LoggingLevel::Info);
        ktt::Logger::GetLogger().SetLoggingTarget(m_Stream);
    }

    ~LogCapture()
    {
        ktt::Logger::GetLogger().SetLoggingTarget(std::clog);
        ktt::Logger::GetLogger().SetLoggingLevel(m_Level);
    }

    std::string Text() const
    {
        return m_Stream.str();
    }

private:
    std::ostringstream m_Stream;
    ktt::LoggingLevel m_Level;
};

std::vector<SourceDefinition> DefinitionsAB()
{
    return {{"a", SourceA}, {"b", SourceB}};
}

ktt::db::TuningInfo TuningInfoAB()
{
    auto info = MakeTuningInfo();
    info.spaceInfo.sourceFingerprint = FingerprintAB;
    return info;
}

std::string CodeUrl(const size_t fingerprint)
{
    return AppUrl + "/source/" + std::to_string(fingerprint) + "/code";
}

std::string ExceptionText(const std::function<void()> &function)
{
    try
    {
        function();
    }
    catch (const ktt::KttException &error)
    {
        return error.what();
    }

    FAIL("no KttException thrown");
    return {};
}

// SaveSourceCode with the definitions under their own fingerprint, so that only the checks of the definitions apply.
std::string RejectionText(const RemoteDatabase &db, const std::vector<SourceDefinition> &definitions)
{
    return ExceptionText([&] { db.SaveSourceCode(GetSourceFingerprint(definitions), definitions); });
}

bool Contains(const std::string &text, const std::string &part)
{
    return text.find(part) != std::string::npos;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------
// SaveSourceCode (R6.1)

TEST_CASE("SaveSourceCode puts the definitions in their order", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    fixture.http->EnqueueResponse(201, CodeResponse);
    const auto db = fixture.MakeDatabase();

    REQUIRE(db.SaveSourceCode(FingerprintAB, DefinitionsAB()));

    const auto requests = fixture.Requests();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests[0].method == "PUT");
    REQUIRE(requests[0].url == CodeUrl(FingerprintAB));
    REQUIRE(requests[0].url == AppUrl + "/source/16551293918983522932/code");
    REQUIRE(requests[0].Header("x-api-key") == ApiKey);
    REQUIRE(requests[0].Header("correlation-id").has_value());
    REQUIRE(requests[0].Header("content-type") == "application/json");
    REQUIRE(json::parse(requests[0].body) == json{{"definitions", json::array({
        json{{"name", "a"}, {"source", SourceA}}, json{{"name", "b"}, {"source", SourceB}}})}});
}

TEST_CASE("SaveSourceCode returns true for 201 and false for 200", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();

    fixture.http->EnqueueResponse(201, CodeResponse);
    REQUIRE(db.SaveSourceCode(FingerprintAB, DefinitionsAB()));

    // Already stored: also the answer to a repeated upload with other names.
    fixture.http->EnqueueResponse(200, CodeResponse);
    REQUIRE_FALSE(db.SaveSourceCode(FingerprintAB, {{"other", SourceA}, {"", SourceB}}));

    REQUIRE(fixture.Requests().size() == 2);
}

TEST_CASE("SaveSourceCode sends the sources byte for byte", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();

    SECTION("CRLF and trailing spaces are kept")
    {
        const std::vector<SourceDefinition> definitions{{"k", "int a;\r\nint b;  \r\n"}};
        REQUIRE(GetSourceFingerprint(definitions) == 3925580823608318575ULL);
        fixture.http->EnqueueResponse(201, CodeResponse);
        db.SaveSourceCode(3925580823608318575ULL, definitions);

        REQUIRE(fixture.Requests().at(0).body == R"({"definitions":[{"name":"k","source":"int a;\r\nint b;  \r\n"}]})");
    }

    SECTION("UTF-8 is sent as UTF-8, names too")
    {
        const std::string source = "// \xc4\x8d" "esk\xc3\xbd text \xe2\x82\xac \xf0\x9f\x9a\x80\nint x;\n";
        const std::string name = "j\xc3\xa1\xf0\x9f\x9a\x80";
        const std::vector<SourceDefinition> definitions{{name, source}};
        REQUIRE(GetSourceFingerprint(definitions) == 448845483980050019ULL);
        fixture.http->EnqueueResponse(201, CodeResponse);
        db.SaveSourceCode(448845483980050019ULL, definitions);

        REQUIRE(fixture.Requests().at(0).body == R"({"definitions":[{"name":")" + name + R"(","source":"// )"
            + "\xc4\x8d" "esk\xc3\xbd text \xe2\x82\xac \xf0\x9f\x9a\x80" + R"(\nint x;\n"}]})");
    }

    SECTION("an empty name and an empty source are allowed")
    {
        const std::vector<SourceDefinition> definitions{{"", ""}};
        REQUIRE(GetSourceFingerprint(definitions) == 9601993083386333853ULL);
        fixture.http->EnqueueResponse(201, CodeResponse);
        REQUIRE(db.SaveSourceCode(9601993083386333853ULL, definitions));

        REQUIRE(fixture.Requests().at(0).body == R"({"definitions":[{"name":"","source":""}]})");
    }
}

TEST_CASE("SaveSourceCode rejects definitions that do not have the fingerprint", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();

    SECTION("another fingerprint")
    {
        const std::string text = ExceptionText([&db] { db.SaveSourceCode(222, DefinitionsAB()); });
        REQUIRE(Contains(text, Operation));
        REQUIRE(Contains(text, "16551293918983522932"));
        REQUIRE(Contains(text, "222"));
    }

    SECTION("the definitions in another order")
    {
        REQUIRE_THROWS_AS(db.SaveSourceCode(FingerprintAB, {{"b", SourceB}, {"a", SourceA}}), ktt::KttException);
    }

    SECTION("other line endings")
    {
        REQUIRE_THROWS_AS(db.SaveSourceCode(3925580823608318575ULL, {{"k", "int a;\nint b;  \n"}}), ktt::KttException);
    }

    REQUIRE(fixture.Requests().empty());
}

TEST_CASE("SaveSourceCode checks the server's limits before the request", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const size_t MiB = 1024 * 1024;

    SECTION("no definitions")
    {
        REQUIRE(Contains(RejectionText(db, {}), "1 to 32 definitions, got 0"));
    }

    SECTION("33 definitions")
    {
        const std::vector<SourceDefinition> definitions(33, SourceDefinition{"k", "int x;"});
        REQUIRE(Contains(RejectionText(db, definitions), "1 to 32 definitions, got 33"));
    }

    SECTION("a source above 1 MiB")
    {
        const std::string text = RejectionText(db, {{"a", SourceA}, {"big", std::string(MiB + 1, 'x')}});
        REQUIRE(Contains(text, "source of definition 1 has 1048577 bytes"));
    }

    SECTION("sources above 4 MiB together")
    {
        std::vector<SourceDefinition> definitions(4, SourceDefinition{"k", std::string(MiB, 'x')});
        definitions.push_back({"k", "x"});
        REQUIRE(Contains(RejectionText(db, definitions), "4194305 bytes together"));
    }

    SECTION("a name above 1 KiB")
    {
        REQUIRE(Contains(RejectionText(db, {{std::string(1025, 'n'), SourceA}}), "name of definition 0 has 1025 bytes"));
        // A multi-byte character counts with its bytes: 342 x 3 = 1026.
        std::string name;

        for (int i = 0; i < 342; ++i)
            name += "\xe2\x82\xac";

        REQUIRE(Contains(RejectionText(db, {{name, SourceA}}), "1026 bytes"));
    }

    SECTION("invalid UTF-8")
    {
        // A stray continuation byte, an invalid lead byte, an overlong '/', an overlong 3-byte form, a surrogate, a
        // value above U+10FFFF, and a truncated sequence at the end.
        for (const std::string invalid : {"\x80", "\xff", "\xc0\xaf", "\xe0\x80\xaf", "\xed\xa0\x80", "\xf4\x90\x80\x80",
            "\xe2\x82"})
        {
            CAPTURE(invalid);
            const std::string text = RejectionText(db, {{"a", SourceA}, {"b", "int x; " + invalid}});
            REQUIRE(Contains(text, "source of definition 1 is not valid UTF-8 (at byte 7)"));
        }

        REQUIRE(Contains(RejectionText(db, {{"\xc3", SourceA}}), "name of definition 0 is not valid UTF-8"));
    }

    SECTION("U+0000")
    {
        REQUIRE(Contains(RejectionText(db, {{"a", std::string("int x;\0", 7)}}), "source of definition 0 contains the "
            "character U+0000 (at byte 6)"));
        REQUIRE(Contains(RejectionText(db, {{std::string("a\0", 2), SourceA}}), "name of definition 0 contains"));
    }

    SECTION("a body above 5 MiB because of its JSON escaping")
    {
        // 4 MiB of quotes keep the source limits, but each quote is sent as \" (8 MiB).
        const std::vector<SourceDefinition> definitions(4, SourceDefinition{"k", std::string(MiB, '"')});
        const std::string text = RejectionText(db, definitions);
        REQUIRE(Contains(text, Operation));
        REQUIRE(Contains(text, "request body has 8388"));
        REQUIRE(Contains(text, "(5242880)"));
    }

    REQUIRE(fixture.Requests().empty());
}

TEST_CASE("SaveSourceCode accepts code at the limits", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const size_t MiB = 1024 * 1024;

    // 32 definitions, 1 KiB names, 4 MiB of sources in 1 MiB sources, and the edges of UTF-8: U+0080, U+07FF, U+0800,
    // U+D7FF, U+E000, U+FFFF, U+10000 and U+10FFFF.
    std::vector<SourceDefinition> definitions(28, SourceDefinition{std::string(1024, 'n'), "int x;"});
    const std::string edges = "\xc2\x80\xdf\xbf\xe0\xa0\x80\xed\x9f\xbf\xee\x80\x80\xef\xbf\xbf\xf0\x90\x80\x80\xf4\x8f\xbf\xbf";

    for (int i = 0; i < 4; ++i)
    {
        std::string source = edges + std::string(MiB - edges.size() - 6 * (i == 0 ? 28 : 0), 'x');
        definitions.push_back({edges, source});
    }

    REQUIRE(definitions.size() == 32);
    fixture.http->EnqueueResponse(201, CodeResponse);
    REQUIRE(db.SaveSourceCode(GetSourceFingerprint(definitions), definitions));
    REQUIRE(fixture.Requests().size() == 1);
}

TEST_CASE("SaveSourceCode maps error responses to exceptions", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase();
    const auto save = [&db] { db.SaveSourceCode(FingerprintAB, DefinitionsAB()); };

    SECTION("400 with the server's message")
    {
        fixture.http->EnqueueResponse(400, R"({"error":"source code does not match fingerprint 1"})");
        const std::string text = ExceptionText(save);
        REQUIRE(Contains(text, Operation));
        REQUIRE(Contains(text, "HTTP 400"));
        REQUIRE(Contains(text, "PUT /source/16551293918983522932/code"));
        REQUIRE(Contains(text, "source code does not match fingerprint 1"));
    }

    SECTION("404 (no run of the source saved) is an error, not \"nothing\"")
    {
        fixture.http->EnqueueResponse(404, R"({"error":"source not found"})");
        const std::string text = ExceptionText(save);
        REQUIRE(Contains(text, "HTTP 404"));
        REQUIRE(Contains(text, "source not found"));
    }

    SECTION("403 and 413")
    {
        fixture.http->EnqueueResponse(403, R"({"error":"read-only key"})");
        REQUIRE(Contains(ExceptionText(save), "HTTP 403"));
        fixture.http->EnqueueResponse(413, "");
        REQUIRE(Contains(ExceptionText(save), "HTTP 413"));
    }

    SECTION("the API key is never in the message")
    {
        fixture.http->EnqueueResponse(401, R"({"error":"invalid api key )" + ApiKey + R"("})");
        const std::string text = ExceptionText(save);
        REQUIRE_FALSE(Contains(text, ApiKey));
        REQUIRE(Contains(text, "<api key>"));
    }

    SECTION("a network error")
    {
        fixture.http->EnqueueTransportError("connection refused", false);
        const std::string text = ExceptionText(save);
        REQUIRE(Contains(text, Operation));
        REQUIRE(Contains(text, "connection refused"));
    }
}

TEST_CASE("SaveSourceCode is retried after a 503", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const auto db = fixture.MakeDatabase({}, 1);
    fixture.http->EnqueueResponse(503, "");
    fixture.http->EnqueueResponse(201, CodeResponse);

    REQUIRE(db.SaveSourceCode(FingerprintAB, DefinitionsAB()));

    const auto requests = fixture.Requests();
    REQUIRE(requests.size() == 2);
    REQUIRE(requests[0].body == requests[1].body);
}

TEST_CASE("The source fingerprint of SaveSourceCode is the one of the test vectors", "[Database][Remote][SourceCode]")
{
#if defined(_MSC_VER)
    std::ifstream file("../Tests/Database/Utility/SourceFingerprintVectors.json");
#else
    std::ifstream file("../../Tests/Database/Utility/SourceFingerprintVectors.json");
#endif
    REQUIRE(file.is_open());
    const json vectors = json::parse(file);

    for (const auto &vector : vectors.at("vectors"))
    {
        CAPTURE(vector.at("name").get<std::string>());
        std::vector<SourceDefinition> definitions;

        for (const auto &source : vector.at("sources"))
            definitions.push_back({"name", source.get<std::string>()});

        REQUIRE(std::to_string(GetSourceFingerprint(definitions)) == vector.at("fingerprint").get<std::string>());
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The upload after SaveResults (R4.1.8, R6.3)

TEST_CASE("SaveResults uploads the code when the server has none", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const LogCapture log;
    std::vector<size_t> asked;
    const auto db = fixture.MakeDatabase([&asked](const size_t fingerprint) -> std::optional<std::vector<SourceDefinition>>
    {
        asked.push_back(fingerprint);
        return DefinitionsAB();
    });

    fixture.http->EnqueueResponse(201, SaveRunResponse("false"));
    fixture.http->EnqueueResponse(201, CodeResponse);
    db.SaveResults(TuningInfoAB(), {MakeResult("kernel", 64, 3 * Millisecond)});

    REQUIRE(asked == std::vector<size_t>{FingerprintAB});
    const auto requests = fixture.Requests();
    REQUIRE(requests.size() == 2);
    REQUIRE(requests[0].method == "POST");
    REQUIRE(requests[1].method == "PUT");
    REQUIRE(requests[1].url == CodeUrl(FingerprintAB));
    REQUIRE(json::parse(requests[1].body).at("definitions").size() == 2);
    REQUIRE(Contains(log.Text(), "uploaded the code of source 16551293918983522932"));
}

TEST_CASE("SaveResults uploads nothing when it need not", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    int calls = 0;
    const SourceCodeProvider provider = [&calls](size_t) -> std::optional<std::vector<SourceDefinition>>
    {
        ++calls;
        return DefinitionsAB();
    };

    SECTION("the server has the code")
    {
        const auto db = fixture.MakeDatabase(provider);
        fixture.http->EnqueueResponse(201, SaveRunResponse("true"));
        db.SaveResults(TuningInfoAB(), {});
        REQUIRE(calls == 0);
    }

    SECTION("no provider")
    {
        const auto db = fixture.MakeDatabase();
        fixture.http->EnqueueResponse(201, SaveRunResponse("false"));
        db.SaveResults(TuningInfoAB(), {});
    }

    SECTION("the provider does not know the source")
    {
        const LogCapture log;
        const auto db = fixture.MakeDatabase([](size_t) { return std::optional<std::vector<SourceDefinition>>(); });
        fixture.http->EnqueueResponse(201, SaveRunResponse("false"));
        db.SaveResults(TuningInfoAB(), {});
        REQUIRE(Contains(log.Text(), "source-code provider has none to upload"));
    }

    SECTION("the response does not say whether the server has the code")
    {
        const LogCapture log;
        const auto db = fixture.MakeDatabase(provider);
        fixture.http->EnqueueResponse(201, SaveRunResponse(nullptr));
        db.SaveResults(TuningInfoAB(), {});
        REQUIRE(calls == 0);
        REQUIRE(Contains(log.Text(), "does not say whether the server has the code of source 16551293918983522932"));
    }

    REQUIRE(fixture.Requests().size() == 1);
}

TEST_CASE("A failed upload does not fail SaveResults", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    const LogCapture log;

    SECTION("the server rejects the code")
    {
        const auto db = fixture.MakeDatabase([](size_t) { return std::optional(DefinitionsAB()); });
        fixture.http->EnqueueResponse(201, SaveRunResponse("false"));
        fixture.http->EnqueueResponse(500, "upstream said: " + ApiKey);
        REQUIRE_NOTHROW(db.SaveResults(TuningInfoAB(), {}));

        REQUIRE(fixture.Requests().size() == 2);
        REQUIRE(Contains(log.Text(), "the results were saved, but uploading the code of source 16551293918983522932 "
            "failed"));
        REQUIRE(Contains(log.Text(), "HTTP 500"));
        REQUIRE_FALSE(Contains(log.Text(), ApiKey));
    }

    SECTION("the provider returns other code")
    {
        const auto db = fixture.MakeDatabase([](size_t) { return std::optional(std::vector<SourceDefinition>{{"a",
            SourceA}}); });
        fixture.http->EnqueueResponse(201, SaveRunResponse("false"));
        REQUIRE_NOTHROW(db.SaveResults(TuningInfoAB(), {}));

        REQUIRE(fixture.Requests().size() == 1);
        REQUIRE(Contains(log.Text(), "failed: " + Operation));
    }

    SECTION("the provider throws")
    {
        const auto db = fixture.MakeDatabase([](size_t) -> std::optional<std::vector<SourceDefinition>>
        {
            throw std::runtime_error("provider broke");
        });
        fixture.http->EnqueueResponse(201, SaveRunResponse("false"));
        REQUIRE_NOTHROW(db.SaveResults(TuningInfoAB(), {}));
        REQUIRE(Contains(log.Text(), "provider broke"));
    }

    SECTION("the provider throws something else")
    {
        const auto db = fixture.MakeDatabase([](size_t) -> std::optional<std::vector<SourceDefinition>> { throw 42; });
        fixture.http->EnqueueResponse(201, SaveRunResponse("false"));
        REQUIRE_NOTHROW(db.SaveResults(TuningInfoAB(), {}));
        REQUIRE(Contains(log.Text(), "unknown exception"));
    }
}

TEST_CASE("A failed save uploads no code", "[Database][Remote][SourceCode]")
{
    Fixture fixture;
    int calls = 0;
    const auto db = fixture.MakeDatabase([&calls](size_t) -> std::optional<std::vector<SourceDefinition>>
    {
        ++calls;
        return DefinitionsAB();
    });

    fixture.http->EnqueueResponse(400, R"({"error":"invalid"})");
    REQUIRE_THROWS_AS(db.SaveResults(TuningInfoAB(), {}), ktt::KttException);
    REQUIRE(calls == 0);
    REQUIRE(fixture.Requests().size() == 1);
}

#endif // KTT_REMOTE_DATABASE
