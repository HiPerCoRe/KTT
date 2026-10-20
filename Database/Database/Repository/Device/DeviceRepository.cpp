#include <Database/Repository/Device/DeviceRepository.h>

namespace ktt::db
{

// Binds version_major, version_minor and extensions of a device_api row starting at index first, storing unset values
// as their placeholders.
static void BindDeviceApiVersionAndExtensions(Statement& statement, const int first, const DeviceApi& deviceApi)
{
    statement.BindInt(first, deviceApi.cudaComputeCapabilityMajor.value_or(NoCudaComputeCapability));
    statement.BindInt(first + 1, deviceApi.cudaComputeCapabilityMinor.value_or(NoCudaComputeCapability));
    statement.BindText(first + 2, deviceApi.extensions.value_or(NoText));
}

size_t DeviceRepository::CreateDeviceInfo(sqlite3* connection, const DbDeviceInfo& device)
{
    Statement statement(connection, R"(
        INSERT INTO device_info
        (name, vendor, type)
        VALUES (?, ?, ?)
    )");

    statement.BindText(1, device.name);
    statement.BindText(2, device.vendor);
    statement.BindText(3, device.type);
    statement.Execute();

    return sqlite3_last_insert_rowid(connection);
}

std::optional<DbDeviceInfo> DeviceRepository::GetDeviceInfo(sqlite3* connection, const DbDeviceInfo& device)
{
    Statement statement(connection, R"(
        SELECT id, name, vendor, type
        FROM device_info
        WHERE name = ? AND vendor = ? AND type = ?
        LIMIT 1
    )");

    statement.BindText(1, device.name);
    statement.BindText(2, device.vendor);
    statement.BindText(3, device.type);

    if (!statement.Step())
        return std::nullopt;

    return DbDeviceInfo::FromRow(statement);
}

size_t DeviceRepository::CreateDeviceApi(sqlite3* connection, const DeviceApi& deviceApi)
{
    Statement statement(connection, R"(
        INSERT INTO device_api
        (compute_api_id, version_major, version_minor, extensions)
        VALUES (?, ?, ?, ?)
    )");

    statement.BindInt(1, static_cast<int>(deviceApi.computeApi));
    BindDeviceApiVersionAndExtensions(statement, 2, deviceApi);
    statement.Execute();

    return sqlite3_last_insert_rowid(connection);
}

std::optional<DeviceApi> DeviceRepository::GetDeviceApi(sqlite3* connection, const DeviceApi& deviceApi)
{
    Statement statement(connection, R"(
        SELECT id, compute_api_id, version_major, version_minor, extensions
        FROM device_api
        WHERE compute_api_id = ? AND version_major = ? AND version_minor = ? AND extensions = ?
        LIMIT 1
    )");

    statement.BindInt(1, static_cast<int>(deviceApi.computeApi));
    BindDeviceApiVersionAndExtensions(statement, 2, deviceApi);

    if (!statement.Step())
        return std::nullopt;

    return DeviceApi::FromRow(statement);
}

std::optional<DeviceApi> DeviceRepository::GetDeviceApiBySimpleQuery(sqlite3* connection, const DeviceApi& deviceApi)
{
    return GetDeviceApi(connection, deviceApi);
}

size_t DeviceRepository::CreateDevice(sqlite3* connection, const DbDevice& device)
{
    Statement statement(connection, R"(
        INSERT INTO device
        (device_info_id, device_api_id, device_identifier, driver_version)
        VALUES (?, ?, ?, ?)
    )");

    statement.BindInt64(1, static_cast<int64_t>(device.deviceInfoId));
    statement.BindInt64(2, static_cast<int64_t>(device.deviceApiId));
    statement.BindText(3, device.deviceIdentifier);
    statement.BindText(4, device.driverVersion);
    statement.Execute();

    return sqlite3_last_insert_rowid(connection);
}

std::optional<DbDevice> DeviceRepository::GetDevice(sqlite3* connection, const DbDevice& device)
{
    Statement statement(connection, R"(
        SELECT id, device_info_id, device_api_id, device_identifier, driver_version
        FROM device
        WHERE device_info_id = ? AND device_api_id = ? AND device_identifier = ? AND driver_version = ?
        LIMIT 1
    )");

    statement.BindInt64(1, static_cast<int64_t>(device.deviceInfoId));
    statement.BindInt64(2, static_cast<int64_t>(device.deviceApiId));
    statement.BindText(3, device.deviceIdentifier);
    statement.BindText(4, device.driverVersion);

    if (!statement.Step())
        return std::nullopt;

    return DbDevice::FromRow(statement);
}

Device DeviceRepository::GetOrCreateDevice(sqlite3* connection, const Device& device)
{
    Device output = device;

    const DbDeviceInfo deviceInfo{device.infoId, device.name, device.vendor, device.type};
    if (auto existingDeviceInfo = GetDeviceInfo(connection, deviceInfo))
        output.infoId = existingDeviceInfo->id;
    else
        output.infoId = CreateDeviceInfo(connection, deviceInfo);

    const DeviceApi deviceApi{
        device.apiId,
        device.computeApi,
        device.extensions,
        device.cudaComputeCapabilityMajor,
        device.cudaComputeCapabilityMinor
    };

    if (auto existingDeviceApi = GetDeviceApi(connection, deviceApi))
        output.apiId = existingDeviceApi->id;
    else
        output.apiId = CreateDeviceApi(connection, deviceApi);

    const DbDevice dbDevice{device.id, *output.infoId, *output.apiId, device.deviceIdentifier, device.driverVersion};
    if (auto existingDevice = GetDevice(connection, dbDevice))
        output.id = existingDevice->id;
    else
        output.id = CreateDevice(connection, dbDevice);

    return output;
}

Device Device::FromDeviceInfo(const DeviceInfo& deviceInfo)
{
    Device device{};
    device.name = deviceInfo.name;
    device.vendor = deviceInfo.vendor;
    device.type = deviceInfo.type;
    device.computeApi = deviceInfo.computeApi;
    device.extensions = deviceInfo.extensions;
    device.cudaComputeCapabilityMajor = deviceInfo.cudaComputeCapabilityMajor;
    device.cudaComputeCapabilityMinor = deviceInfo.cudaComputeCapabilityMinor;
    device.deviceIdentifier = deviceInfo.deviceIdentifier.value_or(NoText);
    device.driverVersion = deviceInfo.driverVersion;
    return device;
}

DbDeviceInfo DbDeviceInfo::FromRow(const Statement& statement)
{
    return DbDeviceInfo{
        statement.GetSizeT(0),
        statement.GetText(1),
        statement.GetText(2),
        statement.GetText(3)
    };
}

DeviceApi DeviceApi::FromRow(const Statement& statement)
{
    const int major = statement.GetInt(2);
    const int minor = statement.GetInt(3);
    const std::string extensions = statement.GetText(4);

    return DeviceApi{
        statement.GetSizeT(0),
        static_cast<ComputeApi>(statement.GetInt(1)),
        extensions.empty() ? std::nullopt : std::optional<std::string>(extensions),
        major == NoCudaComputeCapability ? std::nullopt : std::optional<int>(major),
        minor == NoCudaComputeCapability ? std::nullopt : std::optional<int>(minor)
    };
}

DbDevice DbDevice::FromRow(const Statement& statement)
{
    return DbDevice{
        statement.GetSizeT(0),
        statement.GetSizeT(1),
        statement.GetSizeT(2),
        statement.GetText(3),
        statement.GetText(4)
    };
}

} // namespace ktt::db
