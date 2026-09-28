# Database ERD

Entity-relationship diagram of the SQLite schema created by `Schema::CreateIfNotExists`
(see [Database/Schema/Schema.cpp](Database/Schema/Schema.cpp)).

```mermaid
erDiagram
    compute_api {
        INTEGER id PK
        TEXT name UK "NOT NULL"
    }

    output_format {
        INTEGER id PK
        TEXT name UK "NOT NULL"
    }

    device_info {
        INTEGER id PK "AUTOINCREMENT"
        TEXT name "NOT NULL"
        TEXT vendor "NOT NULL"
        TEXT type "NOT NULL"
        TIMESTAMP created_at "DEFAULT CURRENT_TIMESTAMP"
    }

    device_api {
        INTEGER id PK "AUTOINCREMENT"
        INTEGER compute_api_id FK "NOT NULL"
        INTEGER version_major "NOT NULL DEFAULT -1"
        INTEGER version_minor "NOT NULL DEFAULT -1"
        TEXT extensions "NOT NULL DEFAULT ''"
    }

    device {
        INTEGER id PK "AUTOINCREMENT"
        INTEGER device_info_id FK "NOT NULL"
        INTEGER device_api_id FK "NOT NULL"
        TEXT device_identifier "NOT NULL DEFAULT ''"
        TEXT driver_version "NOT NULL DEFAULT ''"
    }

    tuner {
        INTEGER id PK "AUTOINCREMENT"
        TEXT name "NOT NULL"
        TEXT version "NOT NULL"
    }

    tuning_source {
        INTEGER id PK "AUTOINCREMENT"
        TEXT source_fingerprint UK "NOT NULL"
        TIMESTAMP created_at "DEFAULT CURRENT_TIMESTAMP"
    }

    tuning_space {
        INTEGER id PK "AUTOINCREMENT"
        INTEGER source_id FK "NOT NULL"
        TEXT parameter_fingerprint "NOT NULL"
        TEXT space_fingerprint "NOT NULL"
        TIMESTAMP created_at "DEFAULT CURRENT_TIMESTAMP"
    }

    tuning_run {
        INTEGER id PK "AUTOINCREMENT"
        BLOB guid UK "NOT NULL"
        INTEGER space_id FK "NOT NULL"
        INTEGER device_id FK "NOT NULL"
        INTEGER tuner_id FK "NOT NULL"
        INTEGER output_format_id FK "NOT NULL"
        TIMESTAMP created_at "DEFAULT CURRENT_TIMESTAMP"
        TEXT input_data "nullable"
    }

    tuning_result {
        INTEGER id PK "AUTOINCREMENT"
        INTEGER run_id FK "NOT NULL"
        INTEGER duration "NOT NULL"
        TEXT result "NOT NULL"
    }

    compute_api   ||--o{ device_api    : "compute_api_id"
    device_info   ||--o{ device        : "device_info_id"
    device_api    ||--o{ device        : "device_api_id"
    tuning_source ||--o{ tuning_space  : "source_id"
    tuning_space  ||--o{ tuning_run    : "space_id"
    device        ||--o{ tuning_run    : "device_id"
    tuner         ||--o{ tuning_run    : "tuner_id"
    output_format ||--o{ tuning_run    : "output_format_id"
    tuning_run    ||--o{ tuning_result : "run_id"
```

## Notes

- `compute_api` and `output_format` are seeded lookup tables:
  - `compute_api`: `OpenCL` (1), `CUDA` (2), `Vulkan` (3), `Cpp` (4)
  - `output_format`: `JSON` (1), `JSON_T4` (2), `XML` (3)
- Unique constraints (beyond the marked `UK` columns):
  - `device_info` — unique on `(name, vendor, type)`
  - `device_api` — unique on `(compute_api_id, version_major, version_minor, extensions)`
  - `device` — unique on `(device_info_id, device_api_id, device_identifier, driver_version)`
  - `tuner` — unique on `(name, version)`
  - `tuning_space` — unique on `(space_fingerprint, parameter_fingerprint, source_id)`
- `device` is where a run executed: a physical device (`device_info` model + `device_identifier`) running a given
  compute API (`device_api`) and driver version. A new row appears when any of these changes (another card, a driver
  update, …). Typical comparisons, all on the one `device` row a run points to:
  - exact same environment: `tuning_run.device_id = ?`
  - same model: `device.device_info_id = ?`
  - same model, API and driver on any card: `device_info_id`, `device_api_id` and `driver_version` equal
  - same physical card on any driver: `device_info_id` and `device_identifier` equal
- `device.device_identifier` is the persistent hardware identifier of the device (the device UUID reported by the
  compute API, e.g. `GPU-<uuid>` for CUDA), from `ktt::DeviceInfo::GetDeviceIdentifier()` via
  `ktt::db::DeviceInfo::deviceIdentifier`.
- `device.driver_version` comes from `ktt::DeviceInfo::GetDriverVersion()` (NVML for CUDA, `CL_DRIVER_VERSION` for
  OpenCL, `VkPhysicalDeviceDriverProperties::driverInfo` for Vulkan 1.2+).
- No column that is part of a unique index is nullable, because SQLite treats every `NULL` as distinct in unique
  indexes and would let duplicate rows through. Unknown or not applicable values are stored as placeholders and read
  back as unset (`std::nullopt` / empty string) in `ktt::db::DeviceInfo`:
  - `device_api.version_major` / `version_minor`: `-1` for devices without a CUDA compute capability
  - `device_api.extensions`: `''` for devices without extensions (CUDA, C++)
  - `device.device_identifier`: `''` when the compute API does not expose a UUID (e.g. the C++/CPU backend)
  - `device.driver_version`: `''` when unknown or not applicable (e.g. the C++/CPU backend)
- `tuner` records the tuner that produced a run, from `ktt::db::TuningInfo::tuner` (name is always `KTT`, version is
  `ktt::GetKttVersionString()`).
- A `tuning_run` is uniquely identified across databases by its `guid`, which is what
  `SyncFromFile` uses to skip already-imported runs.
- `SyncFromFile` copies each run together with its `device` (including device identifier and driver version) and
  `tuner`.
