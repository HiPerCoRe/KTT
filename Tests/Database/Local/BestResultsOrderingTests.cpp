#include <catch.hpp>

#if defined(KTT_DATABASE)

#include <Database/Local/Database.h>

#include "../DatabaseTestHelpers.h"

using namespace ktt::db::test;

TEST_CASE("Database returns best results fastest-first and honours the limit", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    const auto tuningInfo = MakeTuningInfo();

    db.SaveResults(tuningInfo, {
        MakeResult("simpleKernel", 32, 8 * Millisecond),
        MakeResult("simpleKernel", 64, 3 * Millisecond),
        MakeResult("simpleKernel", 128, 5 * Millisecond),
    });

    SECTION("All results are returned ordered by kernel duration ascending")
    {
        const auto loaded = db.SimpleGetBestResults(tuningInfo);

        REQUIRE(loaded.size() == 3);
        REQUIRE(loaded[0].GetKernelDuration() == 3 * Millisecond);
        REQUIRE(loaded[1].GetKernelDuration() == 5 * Millisecond);
        REQUIRE(loaded[2].GetKernelDuration() == 8 * Millisecond);
    }

    SECTION("Only the fastest results are returned when a limit is given")
    {
        const auto loaded = db.SimpleGetBestResults(tuningInfo, 2);

        REQUIRE(loaded.size() == 2);
        REQUIRE(loaded[0].GetKernelDuration() == 3 * Millisecond);
        REQUIRE(loaded[1].GetKernelDuration() == 5 * Millisecond);
    }
}

namespace
{

// A result whose total duration (kernel duration + extra duration) orders differently from its kernel duration.
ktt::KernelResult MakeResultWithExtraDuration(const uint64_t blockSize, const ktt::Nanoseconds kernelDuration,
    const ktt::Nanoseconds extraDuration)
{
    auto result = MakeResult("simpleKernel", blockSize, kernelDuration);
    result.SetExtraDuration(extraDuration);
    return result;
}

ktt::db::GetBestResultsQuery MakeQuery(const ktt::db::TuningInfo &tuningInfo, const uint32_t limit)
{
    return {tuningInfo.spaceInfo, std::nullopt, std::nullopt, limit};
}

} // namespace

TEST_CASE("Database GetBestResults orders by kernel duration, not total duration", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    const auto tuningInfo = MakeTuningInfo();

    SECTION("Within one batch of runs")
    {
        // Kernel durations 3, 5, 8 ms; total durations 13, 5, 8 ms.
        db.SaveResults(tuningInfo, {
            MakeResultWithExtraDuration(32, 8 * Millisecond, 0),
            MakeResultWithExtraDuration(64, 3 * Millisecond, 10 * Millisecond),
            MakeResultWithExtraDuration(128, 5 * Millisecond, 0),
        });

        const auto loaded = db.GetBestResults(MakeQuery(tuningInfo, 50));

        REQUIRE(loaded.size() == 3);
        REQUIRE(loaded[0].GetKernelDuration() == 3 * Millisecond);
        REQUIRE(loaded[0].GetTotalDuration() == 13 * Millisecond);
        REQUIRE(loaded[1].GetKernelDuration() == 5 * Millisecond);
        REQUIRE(loaded[2].GetKernelDuration() == 8 * Millisecond);
    }

    SECTION("Across batches of runs, honouring the limit")
    {
        // GetBestResults pages runs in batches of 500, so the 501st run is merged in a second batch. The fastest
        // kernel is in that second batch, but it has the largest total duration.
        db.SaveResults(tuningInfo, {MakeResultWithExtraDuration(32, 5 * Millisecond, 0)});
        for (int run = 0; run < 499; ++run)
            db.SaveResults(tuningInfo, {MakeResultWithExtraDuration(64, 9 * Millisecond, 0)});
        db.SaveResults(tuningInfo, {MakeResultWithExtraDuration(128, 3 * Millisecond, 10 * Millisecond)});

        const auto loaded = db.GetBestResults(MakeQuery(tuningInfo, 2));

        REQUIRE(loaded.size() == 2);
        REQUIRE(loaded[0].GetKernelDuration() == 3 * Millisecond);
        REQUIRE(loaded[0].GetConfiguration().GetPairs()[0].GetValueUint() == 128);
        REQUIRE(loaded[1].GetKernelDuration() == 5 * Millisecond);
        REQUIRE(loaded[1].GetConfiguration().GetPairs()[0].GetValueUint() == 32);
    }
}

TEST_CASE("Database GetBestResults orders JSON_T4 results by the stored kernel duration", "Database")
{
    // JSON_T4 keeps only the total duration, so a loaded result's GetKernelDuration is the original total duration.
    // Both methods still order by the stored kernel duration (tuning_result.duration), so they agree.
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    const auto tuningInfo = MakeTuningInfo();
    const ktt::db::SaveOptions options{ktt::OutputFormat::JSON_T4, 2};

    SECTION("Within one batch of runs")
    {
        // Kernel durations 3, 5 ms; total durations 13, 5 ms.
        db.SaveResults(tuningInfo, {
            MakeResultWithExtraDuration(32, 3 * Millisecond, 10 * Millisecond),
            MakeResultWithExtraDuration(64, 5 * Millisecond, 0),
        }, options);

        const auto simple = db.SimpleGetBestResults(tuningInfo);
        const auto loaded = db.GetBestResults(MakeQuery(tuningInfo, 50));

        REQUIRE(simple.size() == 2);
        REQUIRE(loaded.size() == 2);
        REQUIRE(loaded[0].GetConfiguration().GetPairs()[0].GetValueUint() == 32);
        REQUIRE(loaded[1].GetConfiguration().GetPairs()[0].GetValueUint() == 64);
        for (size_t i = 0; i < loaded.size(); ++i)
            REQUIRE(loaded[i].GetConfiguration().GetPairs()[0].GetValueUint()
                == simple[i].GetConfiguration().GetPairs()[0].GetValueUint());
    }

    SECTION("Across batches of runs, honouring the limit")
    {
        // The second batch holds the fastest kernel (3 ms), which has the largest total duration (13 ms).
        db.SaveResults(tuningInfo, {MakeResultWithExtraDuration(32, 5 * Millisecond, 0)}, options);
        for (int run = 0; run < 499; ++run)
            db.SaveResults(tuningInfo, {MakeResultWithExtraDuration(64, 9 * Millisecond, 0)}, options);
        db.SaveResults(tuningInfo, {MakeResultWithExtraDuration(128, 3 * Millisecond, 10 * Millisecond)}, options);

        const auto loaded = db.GetBestResults(MakeQuery(tuningInfo, 2));

        REQUIRE(loaded.size() == 2);
        REQUIRE(loaded[0].GetConfiguration().GetPairs()[0].GetValueUint() == 128);
        REQUIRE(loaded[1].GetConfiguration().GetPairs()[0].GetValueUint() == 32);
    }
}

#endif // KTT_DATABASE
