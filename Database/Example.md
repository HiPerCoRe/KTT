# Database usage example

This walks through how `ktt::db::Database` is actually used, following the
end-to-end example in
[/Examples/CoulombSum3dDatabase/CoulombSum3dDatabase.cpp](Examples/CoulombSum3dDatabase/CoulombSum3dDatabase.cpp).

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

### Describe the tuning problem

Needed for getting and saving results. `GetDatabaseTuningInfo` produces the
`TuningInfo` that keys those operations — it carries the source fingerprint,
tuning-space info and device info:

```cpp
const auto tuningInfo = m_tuner.GetDatabaseTuningInfo(m_kernel);
```

## Get stored results

Query the fastest stored results for a tuning problem. The simple form returns
them directly (up to `limit`, default 50):

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
and JSON indentation:

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

## Sync from another database

Needs an open database.
Merges every run from another database file into the local one. Runs are matched
by GUID, so existing runs are skipped — idempotent and safe to repeat. In the
example the path comes from the `--db-sync` CLI option, and the sync only runs when
that option is given:

```cpp
if (!m_dbSyncPath.empty())
{
    db.SyncFromFile(m_dbSyncPath);   // returns the number of runs newly added
}
```

## Public API reference

From [Database/Database.h](Database/Database.h):

| Method | Purpose |
| --- | --- |
| `Database()` | Open/create the default DB (`~/.local/share/ktt/ktt.db` on Linux, `%LOCALAPPDATA%\ktt\ktt.db` on Windows). |
| `Database(std::filesystem::path)` | Open/create a DB at a custom path. |
| `Database(sqlite3*)` | Adopt an already-open connection (not owned; caller closes it). |
| `SaveResults(source, results, options = {})` | Persist a run's `KernelResult`s. |
| `SimpleGetBestResults(source, limit = 50)` | Fastest stored results for a tuning problem. |
| `GetBestResults(query)` | Best results with device / input-data predicates. |
| `GetStatsForSource(sourceFingerprint)` | Aggregate counts (spaces, devices, runs, results) for a source; `std::nullopt` if unknown. |
| `SyncFromFile(otherDatabasePath)` | Merge runs from another DB by GUID; returns count added. |

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
