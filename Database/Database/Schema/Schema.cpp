#include <stdexcept>
#include <string>

#include <Database/Schema/Schema.h>


namespace ktt::db
{

void Schema::CreateIfNotExists(sqlite3* connection)
{
    const auto createSchemaSQL = R"(
CREATE TABLE IF NOT EXISTS compute_api
(
    id   INTEGER PRIMARY KEY,
    name TEXT    NOT NULL UNIQUE
);

INSERT INTO compute_api (id, name) VALUES (1, 'OpenCL') ON CONFLICT(name) DO NOTHING;
INSERT INTO compute_api (id, name) VALUES (2, 'CUDA') ON CONFLICT(name) DO NOTHING;
INSERT INTO compute_api (id, name) VALUES (3, 'Vulkan') ON CONFLICT(name) DO NOTHING;
INSERT INTO compute_api (id, name) VALUES (4, 'Cpp') ON CONFLICT(name) DO NOTHING;

CREATE TABLE IF NOT EXISTS output_format
(
    id   INTEGER PRIMARY KEY,
    name TEXT    NOT NULL UNIQUE
);

INSERT INTO output_format (id, name) VALUES (1, 'JSON') ON CONFLICT(name) DO NOTHING;
INSERT INTO output_format (id, name) VALUES (2, 'JSON_T4') ON CONFLICT(name) DO NOTHING;
INSERT INTO output_format (id, name) VALUES (3, 'XML') ON CONFLICT(name) DO NOTHING;

CREATE TABLE IF NOT EXISTS device_info
(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    vendor     TEXT    NOT NULL,
    type       TEXT    NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_device_name_vendor_type ON device_info (name, vendor, type);

CREATE TABLE IF NOT EXISTS device_api
(
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    compute_api_id INTEGER NOT NULL REFERENCES compute_api (id),
    version_major  INTEGER NOT NULL DEFAULT -1,
    version_minor  INTEGER NOT NULL DEFAULT -1,
    extensions     TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_device_api_compute_api_id ON device_api (compute_api_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_device_api_full ON device_api (compute_api_id, version_major, version_minor, extensions);

CREATE TABLE IF NOT EXISTS device
(
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    device_info_id    INTEGER NOT NULL REFERENCES device_info (id),
    device_api_id     INTEGER NOT NULL REFERENCES device_api (id),
    device_identifier TEXT    NOT NULL DEFAULT '',
    driver_version    TEXT    NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_device_full ON device (device_info_id, device_api_id, device_identifier, driver_version);

CREATE TABLE IF NOT EXISTS tuner
(
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT    NOT NULL,
    version TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tuner_name_version ON tuner (name, version);

CREATE TABLE IF NOT EXISTS tuning_source
(
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_fingerprint TEXT    NOT NULL,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tuning_source_fingerprints ON tuning_source (source_fingerprint);

CREATE TABLE IF NOT EXISTS tuning_space
(
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id             INTEGER NOT NULL REFERENCES tuning_source (id),
    parameter_fingerprint TEXT    NOT NULL,
    space_fingerprint     TEXT    NOT NULL,
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tuning_space_fingerprints ON tuning_space (space_fingerprint, parameter_fingerprint, source_id);

CREATE TABLE IF NOT EXISTS tuning_run
(
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    guid             BLOB    NOT NULL UNIQUE,
    space_id         INTEGER NOT NULL REFERENCES tuning_space (id),
    device_id        INTEGER NOT NULL REFERENCES device (id),
    tuner_id         INTEGER NOT NULL REFERENCES tuner (id),
    output_format_id INTEGER NOT NULL REFERENCES output_format (id),
    created_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    input_data       TEXT
);
CREATE INDEX IF NOT EXISTS idx_tuning_run_space_id ON tuning_run (space_id);
CREATE INDEX IF NOT EXISTS idx_tuning_run_guid ON tuning_run (guid);

CREATE TABLE IF NOT EXISTS tuning_result
(
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL REFERENCES tuning_run (id),
    duration INTEGER NOT NULL,
    result   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tuning_result_run_id ON tuning_result (run_id);

)";

    char* errorMsg = nullptr;

    if (const int result = sqlite3_exec(connection, createSchemaSQL, nullptr, nullptr, &errorMsg); result != SQLITE_OK)
    {
        std::string error = errorMsg ? errorMsg : "Unknown error";
        sqlite3_free(errorMsg);
        throw std::runtime_error("Failed to create table: " + error);
    }
}
} // namespace ktt::db
