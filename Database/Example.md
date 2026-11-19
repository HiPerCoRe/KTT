# Database usage example

This walks through how `ktt::db::Database` is actually used, following the
end-to-end example in
[/Examples/Database/CoulombSum3dLocal/CoulombSum3dLocal.cpp](../Examples/Database/CoulombSum3dLocal/CoulombSum3dLocal.cpp).
The kernel, data and tuning space it shares with the remote variant are in
[/Examples/Database/Common/CoulombSum3dDatabaseExample.h](../Examples/Database/Common/CoulombSum3dDatabaseExample.h).

All database interaction lives in that example's `Run()` method. The snippets
below are grouped by usage — each is an independent operation, not a required
sequence.

## Prerequisites

Build with the database option so `KTT_DATABASE` is defined and the header is
pulled in (see [README.md](README.md)):

```
premake5 gmake --database
```

## Common setup

Opening the database is needed for every operation, including
[syncing](#sync-from-another-database). Describing the tuning problem is only
needed for getting and saving results — syncing works purely on database files
and skips it.

### Open the database

Always needed. The default constructor uses `~/.local/share/ktt/ktt.db` on Linux (`$XDG_DATA_HOME/ktt/ktt.db` when set) and
`%LOCALAPPDATA%\ktt\ktt.db` on Windows,
creating the directory and schema if needed:

```cpp
const ktt::db::Database db;
```

For a custom location, pass a filesystem path instead; an already-open
`sqlite3` connection can also be adopted.

Shown in the remote variant,
[/Examples/Database/CoulombSum3dRemote/CoulombSum3dRemote.cpp](../Examples/Database/CoulombSum3dRemote/CoulombSum3dRemote.cpp)
(it also uses `GetStatsForSource`, the remote-only `QueryResults` and the source code upload): to store the results on a ktt-db server instead,
build with `--remote-database` and open a `RemoteDatabase` with an API key from
the web app's settings, given by the application or read from the file `api-key`
next to the default local database. The server is fixed at build time
(`--remote-database-host`, default `https://supercuda.fi.muni.cz:8443`, reachable only on the FI MUNI VPN). The key decides the realms; see
[README.md](README.md#remote-database). The rest of the example works the same
with it, apart from the differences listed there (for example, `SyncFrom` is
local only, and a limit above 1000 throws).

```cpp
#include <Database/Remote/RemoteDatabase.h>

const ktt::db::RemoteDatabase db;   // reads ~/.local/share/ktt/api-key (throws if it is missing or empty)

// or, with the key from the application (keep it out of the source code):
const char* apiKey = std::getenv("KTT_DB_API_KEY");
const ktt::db::RemoteDatabase db2(apiKey != nullptr ? apiKey : "");   // throws if empty
```

Both classes implement `ktt::db::IDatabase`, so a function that takes a
`const ktt::db::IDatabase&` works with either.

### Describe the tuning problem

Needed for getting and saving results. `GetDatabaseTuningInfo` produces the
`TuningInfo` that keys those operations — it carries the source fingerprint,
tuning-space info and device info:

```cpp
const auto tuningInfo = m_tuner.GetDatabaseTuningInfo(m_kernel);
```

## Get stored results

Query the fastest stored results for a tuning problem. The simple form returns
them directly (up to `limit`, default 50), fastest stored kernel duration first. It
uses results from the same device model, and on CUDA also from other models
with the same compute capability major version and a minor version at most one
apart (see [README.md](README.md#load-results-and-stats)):

```cpp
const auto simpleResults = db.SimpleGetBestResults(tuningInfo, 50);
```

For finer control, `GetBestResults` takes a `GetBestResultsQuery` with optional
device and input-data predicates. Here it restricts to CUDA devices:

```cpp
const ktt::db::GetBestResultsQuery query{
    tuningInfo.spaceInfo,
    std::function<bool(const ktt::db::DeviceInfo &)>([](const ktt::db::DeviceInfo &device) {
        return device.computeApi == ktt::ComputeApi::CUDA;
    }),
    std::nullopt, // no input-data filter
    50            // limit
};
const auto queryResults = db.GetBestResults(query);

std::cout << "Loaded " << simpleResults.size() << " best result(s) from the database "
          << "(" << queryResults.size() << " via CUDA-filtered query)." << std::endl;
```

Loading reads the stored durations in the time unit the results were saved
with, so the loaded durations are right whatever unit `m_tuner.SetTimeUnit` sets
when loading (see [README.md](README.md#time-unit)).

A retrieved `KernelResult` can be turned back into a configuration and executed
directly, avoiding a full tuning sweep when a good config is already known.
Running a configuration is a KTT operation (`m_tuner.Run`), not a database one —
the database only supplies the configuration:

```cpp
if (!simpleResults.empty())
{
    const auto bestConfig = simpleResults[0].GetConfiguration();
    const auto bestResult = m_tuner.Run(m_kernel, bestConfig, {});
    std::cout << "Re-ran best known configuration: "
              << bestResult.GetTotalDuration() << " ns" << std::endl;
}
```

## Tune and save the results

The results you save come from a tuning run. Tuning (`m_tuner.Tune`) is a KTT
operation, not part of the database — `SaveResults` simply persists the results
it produced, so a tuning run must happen first. `inputData` attaches free-form
run metadata (later filterable via the `inputPredicate` in a
`GetBestResultsQuery`); `SaveOptions` controls the stored serialization format
and JSON indentation. The run also stores the time unit active in KTT, which is
used to read the durations back:

```cpp
const auto results = m_tuner.Tune(m_kernel, std::move(m_config->stopCondition),
                                  m_config->preciseParams);

auto save = m_tuner.GetDatabaseTuningInfo(m_kernel);
save.inputData = "atoms=" + std::to_string(m_numberOfAtoms)
               + ";gridSize=" + std::to_string(m_gridWidth);
db.SaveResults(save, results, {ktt::OutputFormat::JSON, 2});
```

The trailing `{ktt::OutputFormat::JSON, 2}` is the `SaveOptions`. Its two
fields — the output format and the JSON indentation — look like this:

```json
{
  "format": "JSON",
  "indent": 2
}
```

### Name the source and the tuning space

Not exercised by the example. Before saving, the source and the tuning space can
be given a human-readable name. Both are optional and unset by default:

```cpp
save.spaceInfo.sourceName = "Coulomb sum 3D";
save.spaceInfo.spaceName = "Default search space";
```

A name is stored only on the first save that creates the source or space; a
later save with the same fingerprints never changes it. The source name comes
back in `SourceStats::name` (see [Reading stats for a source](#reading-stats-for-a-source)).

## Sync from another database

Needs an open database.
Merges every run from another database file into the local one, with each
run's time unit. Runs are matched by GUID, so existing runs are skipped —
idempotent and safe to repeat. The other file is only read by the sync, but
opening it as a `Database` creates its schema when missing, so it must be
writable. In the example the path comes from the
`--db-sync` CLI option, and the sync only runs when that option is given:

```cpp
if (!m_dbSyncPath.empty())
{
    db.SyncFrom(ktt::db::Database(m_dbSyncPath));   // returns the number of runs newly added
}
```

## Public API reference

From [Database/Local/Database.h](Database/Local/Database.h):

| Method | Purpose |
| --- | --- |
| `Database()` | Open/create the default DB (`~/.local/share/ktt/ktt.db` on Linux, `%LOCALAPPDATA%\ktt\ktt.db` on Windows). |
| `Database(std::filesystem::path)` | Open/create a DB at a custom path. |
| `Database(sqlite3*)` | Adopt an already-open connection (not owned; caller closes it). |
| `SaveResults(source, results, options = {})` | Persist a run's `KernelResult`s with the active time unit. |
| `SimpleGetBestResults(source, limit = 50)` | Fastest stored results for a tuning problem on a matching device. |
| `GetBestResults(query)` | Best results with device / input-data predicates. |
| `GetStatsForSource(sourceFingerprint)` | Aggregate counts (spaces, devices, runs, results) and the name of a source; `std::nullopt` if unknown. |
| `SyncFrom(const Database&)` | Merge runs from another DB (path, in-memory or connection) by GUID; returns count added. |

`Database` implements [Database/IDatabase.h](Database/IDatabase.h), which has
`SaveResults`, `SimpleGetBestResults`, `GetBestResults` and `GetStatsForSource`
(not `SyncFrom`). With `--remote-database`,
[Database/Remote/RemoteDatabase.h](Database/Remote/RemoteDatabase.h) adds:

| Method | Purpose |
| --- | --- |
| `RemoteDatabase(options = {})`, `RemoteDatabase(apiKey, options = {})` | Client for the ktt-db server fixed at build time (`GetHost()`); the first reads the API key from the key file (`GetDefaultApiKeyPath()`). Does not contact the server. `RemoteDatabaseOptions` sets timeouts, retries and TLS. |
| `SaveResults`, `SimpleGetBestResults`, `GetBestResults`, `GetStatsForSource` | As for `Database`, on the server; the differences are listed in [README.md](README.md#remote-database). |
| `QueryResults(query)` | Remote only: one page of the results matching a filter over everything the API key can see, with run, device and source; see [README.md](README.md#result-query-remote-only). |
| `SaveSourceCode(sourceFingerprint, definitions)` | Upload the kernel source code for the web app (from `Tuner::GetKernelSources`); `true` if stored now, `false` if the server had it. See [README.md](README.md#kernel-source-code). |
| `MakeSourceCodeProvider(tuner, kernels)` | A `RemoteDatabaseOptions::sourceCodeProvider`, so that `SaveResults` uploads the code of these kernels when the server has none. |

### Reading stats for a source

Not exercised by the example, but shown in [README.md](README.md):

```cpp
const auto stats = db.GetStatsForSource(tuningInfo.spaceInfo.sourceFingerprint);
if (stats)
{
    const nlohmann::json statsJson = *stats;
    std::cout << statsJson.dump(2) << std::endl;
}
```

The JSON holds the counts and the source name (`null` when the source has none):

```json
{
  "deviceCount": 1,
  "name": "Coulomb sum 3D",
  "resultCount": 120,
  "runCount": 3,
  "spaceCount": 1
}
```
