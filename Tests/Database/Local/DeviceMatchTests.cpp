#include <catch.hpp>

#if defined(KTT_DATABASE)

#include <cstdint>
#include <string>

#include <Database/Local/Database.h>

#include "../DatabaseTestHelpers.h"

using namespace ktt::db::test;

namespace
{

// A GPU of the given model on OpenCL or Vulkan, which carry an extensions string and no compute capability.
ktt::db::TuningInfo MakeExtensionsDevice(const std::string &name, const ktt::ComputeApi api)
{
    auto info = MakeTuningInfo();
    info.device.name = name;
    info.device.type = "GPU";
    info.device.computeApi = api;
    info.device.extensions = "ext_a ext_b";
    return info;
}

// A CUDA GPU of the given model and compute capability.
ktt::db::TuningInfo MakeCudaDevice(const std::string &name, const uint32_t major, const uint32_t minor)
{
    auto info = MakeTuningInfo();
    info.device.name = name;
    info.device.type = "GPU";
    info.device.computeApi = ktt::ComputeApi::CUDA;
    info.device.cudaComputeCapabilityMajor = major;
    info.device.cudaComputeCapabilityMinor = minor;
    return info;
}

// Number of stored results SimpleGetBestResults finds for the given device.
size_t CountMatches(const ktt::db::Database &db, const ktt::db::TuningInfo &device)
{
    return db.SimpleGetBestResults(device).size();
}

} // namespace

TEST_CASE("Database matches OpenCL and Vulkan devices by model only", "Database")
{
    const auto api = GENERATE(ktt::ComputeApi::OpenCL, ktt::ComputeApi::Vulkan);
    CAPTURE(static_cast<int>(api));

    const ktt::db::Database db(std::filesystem::path(":memory:"));
    db.SaveResults(MakeExtensionsDevice("GPU A", api), {MakeResult("simpleKernel", 64, 5 * Millisecond)});

    SECTION("The same model matches")
    {
        REQUIRE(CountMatches(db, MakeExtensionsDevice("GPU A", api)) == 1);
    }

    SECTION("Another model with the same compute API and extensions does not match")
    {
        REQUIRE(CountMatches(db, MakeExtensionsDevice("GPU B", api)) == 0);
    }

    SECTION("Another vendor or type of the same name does not match")
    {
        auto otherVendor = MakeExtensionsDevice("GPU A", api);
        otherVendor.device.vendor = "Other Vendor";
        REQUIRE(CountMatches(db, otherVendor) == 0);

        auto otherType = MakeExtensionsDevice("GPU A", api);
        otherType.device.type = "Accelerator";
        REQUIRE(CountMatches(db, otherType) == 0);
    }

    SECTION("The same model with other extensions matches")
    {
        auto otherExtensions = MakeExtensionsDevice("GPU A", api);
        otherExtensions.device.extensions = "ext_c";
        REQUIRE(CountMatches(db, otherExtensions) == 1);
    }
}

TEST_CASE("Database matches C++ devices by model only", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    db.SaveResults(MakeTuningInfo(), {MakeResult("simpleKernel", 64, 5 * Millisecond)});

    REQUIRE(CountMatches(db, MakeTuningInfo()) == 1);

    auto otherModel = MakeTuningInfo();
    otherModel.device.name = "Other CPU";
    REQUIRE(CountMatches(db, otherModel) == 0);
}

TEST_CASE("Database matches CUDA devices by compute capability major and minor within one", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    db.SaveResults(MakeCudaDevice("GPU A", 8, 6), {MakeResult("simpleKernel", 64, 5 * Millisecond)});

    SECTION("Another model with a compute capability minor within one matches")
    {
        REQUIRE(CountMatches(db, MakeCudaDevice("GPU B", 8, 5)) == 1);
        REQUIRE(CountMatches(db, MakeCudaDevice("GPU B", 8, 6)) == 1);
        REQUIRE(CountMatches(db, MakeCudaDevice("GPU B", 8, 7)) == 1);
    }

    SECTION("Another model with a compute capability minor two or more away does not match")
    {
        REQUIRE(CountMatches(db, MakeCudaDevice("GPU B", 8, 4)) == 0);
        REQUIRE(CountMatches(db, MakeCudaDevice("GPU B", 8, 8)) == 0);
    }

    SECTION("Another model with another compute capability major does not match")
    {
        REQUIRE(CountMatches(db, MakeCudaDevice("GPU B", 7, 6)) == 0);
        REQUIRE(CountMatches(db, MakeCudaDevice("GPU B", 9, 6)) == 0);
    }

    SECTION("The same model matches whatever its compute capability")
    {
        REQUIRE(CountMatches(db, MakeCudaDevice("GPU A", 9, 0)) == 1);
    }
}

TEST_CASE("Database does not match devices across compute APIs by capability", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    db.SaveResults(MakeCudaDevice("GPU A", 8, 6), {MakeResult("simpleKernel", 64, 5 * Millisecond)});
    db.SaveResults(MakeExtensionsDevice("GPU C", ktt::ComputeApi::OpenCL), {MakeResult("simpleKernel", 64, 7 * Millisecond)});

    SECTION("An OpenCL query does not match a stored CUDA device of another model")
    {
        REQUIRE(CountMatches(db, MakeExtensionsDevice("GPU B", ktt::ComputeApi::OpenCL)) == 0);
    }

    SECTION("A CUDA query does not match a stored OpenCL device of another model")
    {
        const auto loaded = db.SimpleGetBestResults(MakeCudaDevice("GPU B", 8, 6));
        REQUIRE(loaded.size() == 1);
        REQUIRE(loaded[0].GetKernelDuration() == 5 * Millisecond);
    }

    SECTION("The same model matches across compute APIs")
    {
        const auto loaded = db.SimpleGetBestResults(MakeExtensionsDevice("GPU A", ktt::ComputeApi::Vulkan));
        REQUIRE(loaded.size() == 1);
        REQUIRE(loaded[0].GetKernelDuration() == 5 * Millisecond);
    }
}

#endif // KTT_DATABASE
