#include <catch.hpp>

#if defined(KTT_DATABASE)

#include <optional>
#include <sqlite3.h>
#include <string>

#include "DatabaseTestHelpers.h"

using namespace ktt::db::test;

namespace
{

// Opens a private in-memory SQLite connection. The Database adopts it without owning it, so the test can inspect
// the stored rows directly through the same connection.
struct InMemoryConnection
{
    sqlite3 *handle = nullptr;

    InMemoryConnection()
    {
        sqlite3_open(":memory:", &handle);
    }

    ~InMemoryConnection()
    {
        sqlite3_close(handle);
    }
};

// Runs a query returning a single integer (e.g. a COUNT).
int64_t QueryInt(sqlite3 *connection, const std::string &sql)
{
    sqlite3_stmt *stmt = nullptr;
    sqlite3_prepare_v2(connection, sql.c_str(), -1, &stmt, nullptr);
    REQUIRE(sqlite3_step(stmt) == SQLITE_ROW);
    const int64_t value = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    return value;
}

// Runs a query returning a single text value.
std::string QueryText(sqlite3 *connection, const std::string &sql)
{
    sqlite3_stmt *stmt = nullptr;
    sqlite3_prepare_v2(connection, sql.c_str(), -1, &stmt, nullptr);
    REQUIRE(sqlite3_step(stmt) == SQLITE_ROW);
    const std::string value = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
    sqlite3_finalize(stmt);
    return value;
}

// Loads the best results through the query API and captures the DeviceInfo passed to the device predicate.
std::optional<ktt::db::DeviceInfo> LoadDeviceInfo(const ktt::db::Database &db, const ktt::db::TuningInfo &tuningInfo)
{
    std::optional<ktt::db::DeviceInfo> seen;
    const ktt::db::GetBestResultsQuery query{
        tuningInfo.spaceInfo,
        std::function<bool(const ktt::db::DeviceInfo &)>([&seen](const ktt::db::DeviceInfo &device) {
            seen = device;
            return true;
        }),
        std::nullopt,
        50
    };
    db.GetBestResults(query);
    return seen;
}

} // namespace

TEST_CASE("Database stores the driver version and tuner of a run", "Database")
{
    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle);

    auto tuningInfo = MakeTuningInfo();
    tuningInfo.device.driverVersion = "550.54.14";
    tuningInfo.tuner = {"KTT", "2.3.1"};

    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)});

    const auto device = LoadDeviceInfo(db, tuningInfo);
    REQUIRE(device.has_value());
    REQUIRE(device->driverVersion == "550.54.14");

    REQUIRE(
        QueryText(
            connection.handle,
            "SELECT tuner.name FROM tuning_run JOIN tuner ON tuner.id = tuning_run.tuner_id"
        ) == "KTT"
    );
    REQUIRE(
        QueryText(
            connection.handle,
            "SELECT tuner.version FROM tuning_run JOIN tuner ON tuner.id = tuning_run.tuner_id"
        ) == "2.3.1"
    );
}

TEST_CASE("Database defaults the tuner to the current KTT version", "Database")
{
    const auto tuningInfo = MakeTuningInfo();

    REQUIRE(tuningInfo.tuner.name == "KTT");
    REQUIRE(tuningInfo.tuner.version == ktt::GetKttVersionString());
}

TEST_CASE("Database reuses device and tuner rows across runs", "Database")
{
    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle);

    auto tuningInfo = MakeTuningInfo();
    tuningInfo.device.driverVersion = "550.54.14";

    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)});
    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 128, 6 * Millisecond)});

    REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM tuning_run") == 2);
    REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device") == 1);
    REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM tuner") == 1);

    SECTION("A different driver version creates a new device but reuses device_info and device_api")
    {
        tuningInfo.device.driverVersion = "560.28.03";
        db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 4 * Millisecond)});

        REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device") == 2);
        REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device_info") == 1);
        REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device_api") == 1);
    }

    SECTION("A different device identifier creates a new device for the same device_info and device_api")
    {
        tuningInfo.device.deviceIdentifier = "GPU-00000000-0000-0000-0000-000000000001";
        db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 4 * Millisecond)});

        REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device") == 2);
        REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device_info") == 1);
        REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device_api") == 1);
    }

    SECTION("A different tuner version creates a new tuner")
    {
        tuningInfo.tuner.version = "9.9.9";
        db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 4 * Millisecond)});

        REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM tuner") == 2);
        REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device") == 1);
    }
}

TEST_CASE("Database reuses device rows for devices without CUDA compute capability", "Database")
{
    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle);

    auto tuningInfo = MakeTuningInfo();
    tuningInfo.device.type = "GPU";
    tuningInfo.device.computeApi = ktt::ComputeApi::OpenCL;
    tuningInfo.device.extensions = "cl_khr_fp64";

    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)});
    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 128, 6 * Millisecond)});

    REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device_api") == 1);
    REQUIRE(QueryInt(connection.handle, "SELECT COUNT(*) FROM device") == 1);
}

TEST_CASE("Database stores the device identifier on the device", "Database")
{
    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle);

    auto tuningInfo = MakeTuningInfo();
    tuningInfo.device.deviceIdentifier = "GPU-00000000-0000-0000-0000-000000000001";

    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)});

    REQUIRE(
        QueryText(connection.handle, "SELECT device_identifier FROM device") ==
        "GPU-00000000-0000-0000-0000-000000000001"
    );

    const auto device = LoadDeviceInfo(db, tuningInfo);
    REQUIRE(device.has_value());
    REQUIRE(device->deviceIdentifier == "GPU-00000000-0000-0000-0000-000000000001");
}

TEST_CASE("Database reads unset device fields back as unset", "Database")
{
    const ktt::db::Database db(std::filesystem::path(":memory:"));
    const auto tuningInfo = MakeTuningInfo(); // C++ device: no identifier, capability or extensions

    db.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)});

    const auto device = LoadDeviceInfo(db, tuningInfo);
    REQUIRE(device.has_value());
    REQUIRE_FALSE(device->deviceIdentifier.has_value());
    REQUIRE_FALSE(device->cudaComputeCapabilityMajor.has_value());
    REQUIRE_FALSE(device->cudaComputeCapabilityMinor.has_value());
    REQUIRE_FALSE(device->extensions.has_value());
    REQUIRE(device->driverVersion.empty());
}

TEST_CASE("Database unique indexes reject duplicate device rows with unset fields", "Database")
{
    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle); // creates the schema

    // Inserted with all optional columns left at their defaults, which previously were NULL and slipped past the index.
    const char *insertApi = "INSERT INTO device_api (compute_api_id) VALUES (4)";
    REQUIRE(sqlite3_exec(connection.handle, insertApi, nullptr, nullptr, nullptr) == SQLITE_OK);
    REQUIRE(sqlite3_exec(connection.handle, insertApi, nullptr, nullptr, nullptr) == SQLITE_CONSTRAINT);

    const char *insertInfo = "INSERT INTO device_info (name, vendor, type) VALUES ('CPU', 'Vendor', 'CPU')";
    REQUIRE(sqlite3_exec(connection.handle, insertInfo, nullptr, nullptr, nullptr) == SQLITE_OK);

    const char *insertDevice = "INSERT INTO device (device_info_id, device_api_id) VALUES (1, 1)";
    REQUIRE(sqlite3_exec(connection.handle, insertDevice, nullptr, nullptr, nullptr) == SQLITE_OK);
    REQUIRE(sqlite3_exec(connection.handle, insertDevice, nullptr, nullptr, nullptr) == SQLITE_CONSTRAINT);
}

TEST_CASE("Database sync copies the device identifier, driver version and tuner", "Database")
{
    const auto path = std::filesystem::temp_directory_path() / "ktt_database_test_sync_device_tuner.db";
    std::filesystem::remove(path);

    auto tuningInfo = MakeTuningInfo();
    tuningInfo.device.driverVersion = "550.54.14";
    tuningInfo.device.deviceIdentifier = "GPU-00000000-0000-0000-0000-000000000001";
    tuningInfo.tuner = {"KTT", "2.3.1"};

    {
        const ktt::db::Database other(path);
        other.SaveResults(tuningInfo, {MakeResult("simpleKernel", 64, 5 * Millisecond)});
    }

    InMemoryConnection connection;
    const ktt::db::Database db(connection.handle);
    REQUIRE(db.SyncFrom(ktt::db::Database(path)) == 1);

    const auto device = LoadDeviceInfo(db, tuningInfo);
    REQUIRE(device.has_value());
    REQUIRE(device->driverVersion == "550.54.14");
    REQUIRE(device->deviceIdentifier == "GPU-00000000-0000-0000-0000-000000000001");

    REQUIRE(
        QueryText(
            connection.handle,
            "SELECT tuner.version FROM tuning_run JOIN tuner ON tuner.id = tuning_run.tuner_id"
        ) == "2.3.1"
    );

    std::filesystem::remove(path);
}

#endif // KTT_DATABASE
