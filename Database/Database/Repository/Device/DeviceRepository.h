#pragma once

#include <Ktt.h>
#include <Api/Info/DatabaseTuningInfo.h>
#include <Database/Utility/Statement.h>
#include <cstddef>
#include <optional>
#include <sqlite3.h>
#include <string>


namespace ktt::db
{

// Unknown or not applicable values are stored as these placeholders instead of NULL, because SQLite treats every NULL
// as distinct in unique indexes, which would let duplicate device and device_api rows through.
inline constexpr int NoCudaComputeCapability = -1; ///< device_api.version_major/minor of non-CUDA devices.
inline constexpr const char* NoText = ""; ///< Empty device_api.extensions, device.device_identifier, device.driver_version.

/** @struct DbDeviceInfo
 * Represents device hardware information stored in the database.
 */
struct DbDeviceInfo
{
    std::optional<size_t> id; ///< Unique database identifier.
    std::string name; ///< Device name.
    std::string vendor; ///< Device vendor.
    std::string type; ///< Device type (e.g., "GPU", "CPU").

    /** @fn static DbDeviceInfo FromRow(const Statement& statement)
     * Creates a DbDeviceInfo from a database query result row.
     * @param statement Statement positioned at a result row.
     * @return DbDeviceInfo populated from the current row.
     */
    static DbDeviceInfo FromRow(const Statement& statement);
};

/** @struct DeviceApi
 * Represents device compute API capabilities and version information.
 */
struct DeviceApi
{
    std::optional<size_t> id; ///< Unique database identifier.
    ComputeApi computeApi; ///< Compute API type (OpenCL, CUDA, etc.).
    std::optional<std::string> extensions; ///< Supported extensions string.
    std::optional<int> cudaComputeCapabilityMajor; ///< CUDA major compute capability.
    std::optional<int> cudaComputeCapabilityMinor; ///< CUDA minor compute capability.

    /** @fn static DeviceApi FromRow(const Statement& statement)
     * Creates a DeviceApi from a database query result row.
     * @param statement Statement positioned at a result row.
     * @return DeviceApi populated from the current row.
     */
    static DeviceApi FromRow(const Statement& statement);
};

/** @struct DbDevice
 * Represents a row of the device table: a physical device (device_info + device identifier) running a given compute
 * API (device_api) and driver version.
 */
struct DbDevice
{
    std::optional<size_t> id; ///< Unique database identifier.
    size_t deviceInfoId{}; ///< Reference to device_info table.
    size_t deviceApiId{}; ///< Reference to device_api table.
    std::string deviceIdentifier; ///< Persistent hardware identifier (UUID), empty when unknown.
    std::string driverVersion; ///< Driver version, empty when unknown.

    /** @fn static DbDevice FromRow(const Statement& statement)
     * Creates a DbDevice from a database query result row.
     * @param statement Statement positioned at a result row.
     * @return DbDevice populated from the current row.
     */
    static DbDevice FromRow(const Statement& statement);
};

/** @struct Device
 * Complete device information combining hardware, compute API and driver details.
 */
struct Device
{
    std::optional<size_t> id; ///< Reference to device table.
    std::optional<size_t> infoId; ///< Reference to device_info table.
    std::optional<size_t> apiId; ///< Reference to device_api table.
    std::string name; ///< Device name.
    std::string vendor; ///< Device vendor.
    std::string type; ///< Device type.
    ComputeApi computeApi; ///< Compute API type.
    std::optional<std::string> extensions; ///< Supported extensions.
    std::optional<int> cudaComputeCapabilityMajor; ///< CUDA major compute capability.
    std::optional<int> cudaComputeCapabilityMinor; ///< CUDA minor compute capability.
    std::string deviceIdentifier; ///< Persistent hardware identifier (UUID), empty when unknown.
    std::string driverVersion; ///< Driver version, empty when unknown.

    /** @fn static Device FromDeviceInfo(const DeviceInfo& deviceInfo)
     * Creates a Device (without database ids) from the public DeviceInfo.
     * @param deviceInfo Device information as carried by TuningInfo.
     * @return Device with all ids unset.
     */
    static Device FromDeviceInfo(const DeviceInfo& deviceInfo);
};

/** @class DeviceRepository
 * Data access layer for device information in the database.
 * Provides methods to create, retrieve, and manage device and device API records.
 */
class DeviceRepository
{
public:
    /** @fn static Device GetOrCreateDevice(sqlite3* connection, const Device& device)
     * Gets or creates a device record in the database.
     * Resolves (or creates) the device_info and device_api records first, then the device record combining them
     * with the driver version.
     * @param connection SQLite database connection.
     * @param device Device information to get or create.
     * @return Device struct with populated id, infoId and apiId fields.
     */
    static Device GetOrCreateDevice(sqlite3* connection, const Device& device);

    /** @fn static size_t CreateDevice(sqlite3* connection, const DbDevice& device)
     * Creates a new device record (device_info + device_api + device identifier + driver version) in the database.
     * @param connection SQLite database connection.
     * @param device Device record to insert.
     * @return Database ID of the newly created device record.
     * @throw KttException If insertion fails.
     */
    static size_t CreateDevice(sqlite3* connection, const DbDevice& device);

    /** @fn static std::optional<DbDevice> GetDevice(sqlite3* connection, const DbDevice& device)
     * Retrieves a device record matching the device_info, device_api, device identifier and driver version.
     * @param connection SQLite database connection.
     * @param device Device record to search for.
     * @return Optional DbDevice if found, std::nullopt otherwise.
     * @throw KttException If query fails.
     */
    static std::optional<DbDevice> GetDevice(sqlite3* connection, const DbDevice& device);

    /** @fn static size_t CreateDeviceInfo(sqlite3* connection, const DbDeviceInfo& device)
     * Creates a new device_info record in the database.
     * @param connection SQLite database connection.
     * @param device Device information to insert.
     * @return Database ID of the newly created device_info record.
     * @throw KttException If insertion fails.
     */
    static size_t CreateDeviceInfo(sqlite3* connection, const DbDeviceInfo& device);

    /** @fn static size_t CreateDeviceApi(sqlite3* connection, const DeviceApi& deviceApi)
     * Creates a new device API record in the database.
     * @param connection SQLite database connection.
     * @param deviceApi Device API information to insert.
     * @return Database ID of the newly created device_api record.
     * @throw KttException If insertion fails.
     */
    static size_t CreateDeviceApi(sqlite3* connection, const DeviceApi& deviceApi);

    /** @fn static std::optional<DbDeviceInfo> GetDeviceInfo(sqlite3* connection, const DbDeviceInfo& device)
     * Retrieves device hardware information from the database.
     * Searches for a device by name, vendor, and type.
     * @param connection SQLite database connection.
     * @param device Device information to search for.
     * @return Optional DbDeviceInfo if found, std::nullopt otherwise.
     * @throw KttException If query fails.
     */
    static std::optional<DbDeviceInfo> GetDeviceInfo(sqlite3* connection, const DbDeviceInfo& device);

    /** @fn static std::optional<DeviceApi> GetDeviceApi(sqlite3* connection, const DeviceApi& deviceApi)
     * Retrieves device API information from the database.
     * Searches for a device API by compute API type and version information. Unset fields match their placeholder.
     * @param connection SQLite database connection.
     * @param deviceApi Device API information to search for.
     * @return Optional DeviceApi if found, std::nullopt otherwise.
     * @throw KttException If query fails.
     */
    static std::optional<DeviceApi> GetDeviceApi(sqlite3* connection, const DeviceApi& deviceApi);

    /** @fn static std::optional<DeviceApi> GetDeviceApiBySimpleQuery(sqlite3* connection, const DeviceApi& deviceApi)
     * Retrieves device API information. Equivalent to GetDeviceApi now that unset fields are stored as placeholders
     * instead of NULL; kept for compatibility.
     * @param connection SQLite database connection.
     * @param deviceApi Device API information to search for.
     * @return Optional DeviceApi if found, std::nullopt otherwise.
     * @throw KttException If query fails.
     */
    static std::optional<DeviceApi> GetDeviceApiBySimpleQuery(sqlite3* connection, const DeviceApi& deviceApi);
};

} // namespace ktt::db
