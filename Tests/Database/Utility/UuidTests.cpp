#include <catch.hpp>

#if defined(KTT_DATABASE) || defined(KTT_REMOTE_DATABASE)

#include <chrono>
#include <cstdint>
#include <set>
#include <string>

#include <Api/KttException.h>
#include <Database/Utility/Uuid.h>

using ktt::db::uuid;
using ktt::db::UuidGenerator;

namespace
{

uint64_t TimestampMs(const uuid& id)
{
    uint64_t ms = 0;

    for (int i = 0; i < 6; ++i)
        ms = (ms << 8) | id[i];

    return ms;
}

uint64_t NowMs()
{
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

} // namespace

TEST_CASE("GenerateUuid produces a version 4 UUID with the RFC variant", "Uuid")
{
    const auto id = UuidGenerator::GenerateUuid();

    REQUIRE(UuidGenerator::GetVersion(id) == 4);
    REQUIRE((id[8] & 0xC0) == 0x80);
}

TEST_CASE("GenerateUuidV7 produces a version 7 UUID with the RFC variant", "Uuid")
{
    const auto id = UuidGenerator::GenerateUuidV7();

    REQUIRE(UuidGenerator::GetVersion(id) == 7);
    REQUIRE((id[8] & 0xC0) == 0x80);
}

TEST_CASE("GenerateUuidV7 embeds the current time in milliseconds", "Uuid")
{
    const auto before = NowMs();
    const auto id = UuidGenerator::GenerateUuidV7();
    const auto after = NowMs();

    REQUIRE(TimestampMs(id) >= before);
    REQUIRE(TimestampMs(id) <= after);
}

TEST_CASE("GenerateUuidV7 ids from different milliseconds sort in creation order", "Uuid")
{
    const auto first = UuidGenerator::GenerateUuidV7();
    const auto start = NowMs();

    while (NowMs() == start)
    {
    }

    const auto second = UuidGenerator::GenerateUuidV7();

    REQUIRE(first < second);
}

TEST_CASE("GenerateUuidV7 does not repeat", "Uuid")
{
    std::set<uuid> ids;

    for (int i = 0; i < 10000; ++i)
        ids.insert(UuidGenerator::GenerateUuidV7());

    REQUIRE(ids.size() == 10000);
}

TEST_CASE("ToString writes the lowercase 8-4-4-4-12 form", "Uuid")
{
    const uuid id{0x01, 0x93, 0xA1, 0xF2, 0x7C, 0x4E, 0x7B, 0x3A, 0x9D, 0x21, 0x5E, 0x8F, 0x0C, 0x6A, 0x4B, 0x17};
    REQUIRE(UuidGenerator::ToString(id) == "0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b17");
}

TEST_CASE("Parse reads the 8-4-4-4-12 form in either case", "Uuid")
{
    const uuid expected{0x01, 0x93, 0xA1, 0xF2, 0x7C, 0x4E, 0x7B, 0x3A, 0x9D, 0x21, 0x5E, 0x8F, 0x0C, 0x6A, 0x4B, 0x17};

    REQUIRE(UuidGenerator::Parse("0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b17") == expected);
    REQUIRE(UuidGenerator::Parse("0193A1F2-7C4E-7B3A-9D21-5E8F0C6A4B17") == expected);
}

TEST_CASE("ToString and Parse round-trip v4 and v7 UUIDs", "Uuid")
{
    for (int i = 0; i < 100; ++i)
    {
        const auto v4 = UuidGenerator::GenerateUuid();
        const auto v7 = UuidGenerator::GenerateUuidV7();

        REQUIRE(UuidGenerator::Parse(UuidGenerator::ToString(v4)) == v4);
        REQUIRE(UuidGenerator::Parse(UuidGenerator::ToString(v7)) == v7);
        REQUIRE(UuidGenerator::GetVersion(UuidGenerator::Parse(UuidGenerator::ToString(v7))) == 7);
    }
}

TEST_CASE("Parse rejects text that is not a UUID", "Uuid")
{
    for (const std::string text : {"", "0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b1", "0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b177",
             "0193a1f207c4e-7b3a-9d21-5e8f0c6a4b17", "0193a1f2-7c4e-7b3a-9d21-5e8f0c6a4bg7",
             "{193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b1}", "0193a1f27c4e7b3a9d215e8f0c6a4b17", " 193a1f2-7c4e-7b3a-9d21-5e8f0c6a4b17"})
    {
        CAPTURE(text);
        REQUIRE_THROWS_AS(UuidGenerator::Parse(text), ktt::KttException);
    }
}

#endif // KTT_DATABASE || KTT_REMOTE_DATABASE
