# Database usage

This module provides a lightweight SQLite-backed store for tuning results. Typical usage is guarded by the KTT_DATABASE define.

For hands-on, code-level usage see [Example.md](Example.md), which walks through the end-to-end example in [/Examples/Database/CoulombSum3dLocal/CoulombSum3dLocal.cpp](../Examples/Database/CoulombSum3dLocal/CoulombSum3dLocal.cpp). The same example against a ktt-db server is [/Examples/Database/CoulombSum3dRemote/CoulombSum3dRemote.cpp](../Examples/Database/CoulombSum3dRemote/CoulombSum3dRemote.cpp). The schema behind these operations is documented in [ERD.md](ERD.md). To run the remote example against the FI MUNI server step by step, see [RemoteQuickStart.md](RemoteQuickStart.md).

## Enable database usage

The local and the remote database are separate premake5 options. Use either one, or both:

| Options | Builds | Defines | Needs |
|---|---|---|---|
| `--database` | the local SQLite `Database` | `KTT_DATABASE` | sqlite3 (bundled) |
| `--remote-database` | the remote client `RemoteDatabase` (see [Remote database](#remote-database)) | `KTT_REMOTE_DATABASE` | libcurl |
| `--database --remote-database` | both | both | both |
| `--remote-database-host=URL` | with `--remote-database`: the server `RemoteDatabase` talks to | `KTT_REMOTE_DATABASE_HOST` | – |

The shared part (`IDatabase.h` and `Utility/`) is built with either option. A local-only build does not link
libcurl, and a remote-only build leaves out `Local/` and sqlite3. The server host is fixed at build time and defaults
to the FI MUNI server, `https://supercuda.fi.muni.cz:8443` (reachable only on the FI MUNI VPN). With `KTT_DATABASE`, `Ktt.h` includes the local
`Database`; the remote client is included with `#include <Database/Remote/RemoteDatabase.h>`.

## Source layout

Headers are included relative to this folder, e.g. `#include <Database/Local/Database.h>`.

```
Database/
  IDatabase.h     the interface both databases implement
  Utility/        shared by both: Uuid, TimeUnitGuard, TuningInfoValidation, ResultSerialization(JsonT4),
                  DataDirectory (the folder of ktt.db and the API key file)
  Local/          the SQLite Database (built with --database only)
    Repository/   one repository per table group (Device, Result, Run, Source, Space, Tuner)
    Schema/
    Utility/      Statement, TransactionGuard, JsonConverters, SqlList
  Remote/         RemoteDatabase (built with --remote-database only)
    Http/         HttpClient and its libcurl implementation
    Utility/      RemoteClient, RemoteCall (request and error mapping), ApiKeyFile, RemoteInternal.h
      Wire/       one type per request and response of the ktt-db API (to_json/from_json and the mapping to KTT's types)
```

The tests in `Tests/Database/` follow the same split: `Local/`, `Remote/` and `Utility/`.

## One interface for local and remote

`Database` (SQLite) implements the interface `ktt::db::IDatabase` (`Database/IDatabase.h`), and so does
`RemoteDatabase`. Code that only saves and reads results can take a `const ktt::db::IDatabase&` and work with either:

```cpp
void SaveRun(const ktt::db::IDatabase& db, const ktt::db::TuningInfo& info, std::vector<ktt::KernelResult> results)
{
    db.SaveResults(info, std::move(results));
}
```

The interface has `SaveResults`, `SimpleGetBestResults`, `GetBestResults` and `GetStatsForSource`, with the same
default arguments on every implementation. `SyncFrom` is on `Database` only, because it copies from one SQLite file to
another.

## Load results and stats

Open a `Database` (default location `~/.local/share/ktt/ktt.db` on Linux, or `$XDG_DATA_HOME/ktt/ktt.db` when set; `%LOCALAPPDATA%\ktt\ktt.db` on Windows) and use `GetDatabaseTuningInfo` to key queries against a tuning problem:

- `SimpleGetBestResults` returns the fastest stored results for that tuning problem on a matching device (see below).
- `GetBestResults` takes a `GetBestResultsQuery` with optional device and input-data predicates for fine-grained filtering.
- `GetStatsForSource` returns aggregate counts (spaces, devices, runs, results) and the name of a source, for a
  source fingerprint.

`SimpleGetBestResults` and `GetBestResults` return the results fastest first, up to the given limit. Both order by the
kernel duration stored when the result was saved (`tuning_result.duration`: `KernelResult::GetKernelDuration` at save
time, in nanoseconds, which leaves out the extra duration of the launcher), so the two methods agree. For results saved
in `JSON_T4`, the loaded `GetKernelDuration` can differ from that key, because the T4 format keeps only the total
duration and loads it as the kernel duration.

`SimpleGetBestResults` uses a stored result when its run's device matches the queried device:

- the same model, that is the same device name, vendor and type, on any compute API; or
- for a CUDA device, a device of another model on CUDA with the same compute capability major version and a minor
  version at most one apart (8.6 matches 8.5, 8.6 and 8.7, but not 8.4, 8.8 or 7.6).

Other device properties (extensions, driver version, device identifier) are not compared. Use `GetBestResults` with a
device predicate for any other rule.

## Time unit

KTT writes the durations inside a serialized result in the time unit active when the result is saved
(`Tuner::SetTimeUnit`, milliseconds by default). `SaveResults` stores that unit with the run (`tuning_run.time_unit`),
and loading a result reads its durations in that stored unit, so the loaded `KernelResult` has the same durations
whatever unit is active when you load it. Runs saved under different units can be mixed in one database, and
`SyncFrom` copies the unit of each run.

To read the stored durations, the database switches KTT's process-wide time unit to the run's unit while it
deserializes a result and restores the previous unit afterwards. `RemoteDatabase` does the same with the unit the
server returns for each result. The database code holds a process-wide lock while it switches the unit, so several
threads that load or save through `Database` or `RemoteDatabase` take turns. The lock does not cover the rest of KTT:
do not load results while another thread uses or changes the time unit outside the database (for example while it
tunes, prints results or calls `Tuner::SetTimeUnit`).

## Save results

Capture results and persist them for later runs with `SaveResults`. You can attach free-form run metadata via `TuningInfo::inputData`, which is later filterable through the input predicate of a `GetBestResultsQuery`.

## Source and space names

A source and a tuning space are identified by their fingerprints. You can also give them a human-readable name
through `TuningInfo::spaceInfo.sourceName` and `spaceName` (both `std::optional<std::string>`, unset by default):

```cpp
auto tuningInfo = tuner.GetDatabaseTuningInfo(kernel);
tuningInfo.spaceInfo.sourceName = "Coulomb sum 3D";
tuningInfo.spaceInfo.spaceName = "Default search space";
db.SaveResults(tuningInfo, results);
```

A name is stored only when the source or space is created, that is on the first save with its fingerprints. A later
save never changes it, even when it carries a different name or none. The source name is returned in
`SourceStats::name` by `GetStatsForSource`.

The schema is not migrated. A database file written before the names existed (without the `name` columns) or before
the time unit was stored (without `tuning_run.time_unit`) is not supported and must be recreated.

## Sync from another database

`SyncFrom` merges all tuning data from another `Database` into the current one. The other database may be
constructed from a path, an in-memory database or an existing `sqlite3*` connection. `SyncFrom` only reads runs from
it, but constructing a `Database` from a path writes to the file: it creates the file and any missing tables and
inserts the compute API rows when they are missing, which needs write access even when nothing changes. So the other
database's file must be writable, also when it is a backup. Runs are matched by their GUID, so runs that already
exist are skipped — the operation is idempotent and safe to repeat. Each newly copied run brings along its tuning
source, tuning space, device and results (the source and space names are applied only when the source or space is new
in the current database), and the copy runs inside a single transaction (a failure leaves the current database
unchanged). Syncing a database from itself (the same connection, or the same file through any path) throws.

```cpp
db.SyncFrom(ktt::db::Database(otherDatabasePath));
```

## Output format and JSON indentation

Database stores tuning results as text in the database. The serialization format and, for JSON, its
indentation are controlled per call through the `SaveOptions` argument to `SaveResults` (not the
constructor). Indentation is useful for SQL reader apps (e.g., DBeaver) that display JSON text more
clearly. `SaveOptions` defaults to JSON with an indent of 2; XML is also supported (indentation is ignored for XML).

If you need a custom database path, construct `Database` with a filesystem path; an already-open `sqlite3` connection can also be adopted.

## Remote database

`ktt::db::RemoteDatabase` (`Database/Remote/RemoteDatabase.h`) stores and reads tuning results on a ktt-db server
instead of a local file. Build KTT with `--remote-database` (see below) and include the header yourself; `Ktt.h` only
includes the local `Database`.

```cpp
#include <Database/Remote/RemoteDatabase.h>

const ktt::db::RemoteDatabase db;           // API key from the key file next to the local database
const ktt::db::RemoteDatabase db2(apiKey);  // API key given by the application
```

- **Server:** fixed when KTT is built, not by the application. The default is the FI MUNI server,
  `https://supercuda.fi.muni.cz:8443`, which is reachable only on the FI MUNI VPN; build with
  `--remote-database-host=https://...` for another server, e.g. `https://localhost:8888` for the local development
  backend.
  `RemoteDatabase::GetHost()` returns it. The client appends `/ktt-db/app/...`. The local backend has a self-signed
  certificate: pass its CA in `RemoteDatabaseOptions::caBundle` (or, for testing only, turn off `verifyTls`).
- **API key:** create one in the web app's settings. A `read` key may only read, a `write` key may also save. **The
  key decides the realms:** reads see what is linked to any of the key's realms, and a save links the new data to all
  of them. There is no realm parameter. KTT never writes the key into exception messages or logs. Give it in one of
  two ways:
  - **by the application:** `RemoteDatabase(apiKey, options = {})`. Keep the key out of source code, e.g. read it from
    an environment variable.
  - **from the key file:** `RemoteDatabase(options = {})` reads the file `api-key` in the same folder as the default
    local database: `~/.local/share/ktt/api-key` on Linux, or `$XDG_DATA_HOME/ktt/api-key` when set;
    `%LOCALAPPDATA%\ktt\api-key` on Windows (`RemoteDatabase::GetDefaultApiKeyPath()`). The file holds only the
    key; spaces and line breaks around it are ignored. A missing or empty file throws, and the message names the
    file. Make the file readable only by you (`chmod 600`).
- **Construction** does not contact the server, so a wrong key shows up on the first call.
- `RemoteDatabase` is not copyable, like `Database`. Its methods are `const` and can be called from several threads
  at once.

`SaveResults`, `SimpleGetBestResults`, `GetBestResults` and `GetStatsForSource` work as described above for the
local `Database`, with these differences:

| Topic | Local `Database` | `RemoteDatabase` |
| --- | --- | --- |
| Errors | `KttException` | `KttException` for every failure: network errors, error statuses (with the server's message and the HTTP status) and malformed responses. An unknown source or space is not an error: the reads return no results, and `GetStatsForSource` returns `std::nullopt`. |
| Output format | JSON, JSON T4 or XML | JSON and JSON T4. A result saved with `OutputFormat::XML` is stored as JSON (a warning is logged once). |
| JSON indentation | `SaveOptions::indent` is applied | Ignored: the server stores the result as a JSON value. |
| `limit` | Any value | At most 1000 for `SimpleGetBestResults` and `GetBestResults`; a larger limit throws without a request. A limit of 0 returns no results without a request. |
| Which data is read | The whole file | Only what is linked to the realms of the API key. `GetStatsForSource` counts only those runs, results, devices and spaces (a space counts when it has a visible run). |
| `SourceStats::name` | The stored source name | Always `std::nullopt`: the server keeps names per realm and does not return them here. |
| Names | Stored when the source or space is created | Sent with every save. The server trims them and stores a name only when the source or space first appears in a realm; a name longer than 100 characters is rejected (`400`). |
| `GetBestResults` | Pages through the runs in the file | Pages through the runs of the space on the server (500 per request), applies the predicates on the client and asks for the best results of the matching runs, so a query over many runs makes several requests. A run that moves to a later page while you read (because another client saved meanwhile) is used once. |
| Runs without input data | Left out by an input predicate | The same. |
| `SyncFrom` | Copies another `Database` | Not available: it is a member of `Database` only, not of `IDatabase`. |
| Save retries | – | A save that fails with a network error or `502` / `503` / `504` is sent again with the same run id, so it is stored once. |
| `QueryResults` | Not available | Remote only, see below. |
| Kernel source code | Not stored | `SaveSourceCode` uploads it, by hand or after each save (see [Kernel source code](#kernel-source-code)). |

The durations of a loaded result are read in the time unit the run was saved with, as locally.

### Result query (remote only)

`QueryResults` searches every result the API key can see, across sources, spaces and runs, with the same filter
language as the web app's query page (`query.md` in the ktt-db docs). It returns one page of results, each with its
run id, input data, device, source fingerprint and (unless `includeResult` is `false`) the stored result.

```cpp
ktt::db::ResultQuery query;
query.filter = "device/compute_api eq 'cuda' and configuration/BLOCK_SIZE ge 32";
query.orderBy = "duration asc";
query.top = 100;

for (const auto& item : db.QueryResults(query))
{
    std::cout << item.runId << " " << item.device.name << " " << item.duration << " ns" << std::endl;
}
```

- **Paging:** `top` (default 50, at most 1000) and `skip` (at most 100000). A larger value throws without a request;
  `top` 0 returns no results without a request. The server always adds `id asc` to the order, so pages are stable.
- **Filter:** sent unchanged. An empty filter matches every result; an invalid one fails with the server's `400`
  message.
- **Durations:** `duration` and `totalDuration` are in nanoseconds; `result` is read in its run's time unit.
- **Configuration values:** a non-negative integer is read as `uint64_t`, a negative one as `int64_t`.
- **Timeouts:** a query the server stops after its time limit fails with `503` and is not retried; narrow the filter.

### Connection options

`RemoteDatabaseOptions` is the optional third constructor argument:

| Field | Default | Meaning |
| --- | --- | --- |
| `connectTimeout` | 5 s | Time limit for establishing the connection (0 = none). |
| `requestTimeout` | 60 s | Time limit for one attempt of a request (0 = none). |
| `maxRetries` | 3 | Additional attempts after a connection error, a timeout or `502` / `503` / `504`. |
| `verifyTls` | `true` | Verify the server certificate and host name. Turn it off only for testing. |
| `caBundle` | none | A CA bundle (PEM file) for a server with a certificate from an internal CA. |
| `sourceCodeProvider` | none | Supplies the kernel source code that `SaveResults` uploads when the server has none (see [Kernel source code](#kernel-source-code)). |

- **Retries** wait 250 ms before the first retry and double the wait each time (up to 8 s). Other statuses (`400`,
  `401`, `403`, `404`, `409`, `500`, ...) are never retried, and neither is a `503` that reports a query timeout,
  because the same query would time out again. Every request the client sends can be repeated safely; a save is
  repeated with the same run id, and the server treats it as the same run.
- **Proxy:** libcurl takes it from the environment. `http_proxy` (for `http://` URLs) is read in lower case only;
  libcurl ignores `HTTP_PROXY` on purpose. `https_proxy`, `all_proxy` and `no_proxy` are read in lower or upper case.
- **Redirects** are not followed, so the key is never sent to another host.

### Building with libcurl

- **Linux:** install the libcurl development package (e.g. `libcurl4-openssl-dev` on Debian and Ubuntu), then
  `premake5 gmake --remote-database`.
- **macOS:** the system libcurl from the SDK is used.
- **Windows (Visual Studio):** set `CURL_ROOT` to a **dynamic** libcurl installation with `include/curl/curl.h` and
  the import library `lib/libcurl.lib` before running premake, e.g. vcpkg's `x64-windows` triplet
  (`vcpkg install curl:x64-windows`, then `CURL_ROOT=<vcpkg>\installed\x64-windows`). The release import library is
  linked in every configuration. Put `libcurl.dll` (and the DLLs it depends on, from the same `bin` folder) next to
  the application. Static libcurl builds (e.g. `x64-windows-static`) and the MinGW packages from curl.se are not
  supported by the build script.

`CURL_ROOT` can also point to a custom libcurl on Linux and macOS.

Only the Linux build has been tested so far; the Windows and macOS setups above have not been built yet.

### Kernel source code

The web app shows the source code of a kernel only when it was uploaded. `SaveSourceCode` uploads it, and
`Tuner::GetKernelSources` returns it, exactly as KTT holds it:

```cpp
db.SaveResults(tuningInfo, results);
db.SaveSourceCode(tuningInfo.spaceInfo.sourceFingerprint, tuner.GetKernelSources(kernel));
```

- **Upload once per source.** The server keeps one copy of the code per source fingerprint, for every realm, and
  never changes it. `SaveSourceCode` returns `true` when it stored the code and `false` when the server already had
  it (also when only the names differ). A run of the source must be saved first; otherwise the server answers `404`
  and `SaveSourceCode` throws.
- **The code must match the fingerprint.** Pass the kernel's definitions unchanged and in their order. The client
  recomputes the source fingerprint and throws before sending anything if it differs; the server checks it again.
  The sources are sent byte for byte (line endings and whitespace included); the names are informational.
- **Limits**, checked before the request: 1 to 32 definitions, each source at most 1 MiB and all of them at most
  4 MiB, each name at most 1 KiB (UTF-8 bytes), valid UTF-8 without the character U+0000, and a request body of at
  most 5 MiB with its JSON escaping.
- **Automatically after a save:** with `RemoteDatabaseOptions::sourceCodeProvider` set, `SaveResults` uploads the
  code when the server answers that it has none for the saved source. `MakeSourceCodeProvider(tuner, kernels)`
  makes a provider that serves the given kernels of a tuner (the tuner must outlive the database). A failed upload
  only logs a warning: the results are saved either way.

```cpp
ktt::db::RemoteDatabaseOptions options;
options.sourceCodeProvider = ktt::db::MakeSourceCodeProvider(tuner, {kernel});
const ktt::db::RemoteDatabase db(options);   // or db(apiKey, options)
```

The test vectors of the source fingerprint, which the server computes the same way, are in
[Tests/Database/Utility/SourceFingerprintVectors.json](../Tests/Database/Utility/SourceFingerprintVectors.json).
