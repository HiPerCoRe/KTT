# Database usage

This module provides a lightweight SQLite-backed store for tuning results. Typical usage is guarded by the KTT_DATABASE define.

For hands-on, code-level usage see [Example.md](Example.md), which walks through the end-to-end example in [/Examples/CoulombSum3dDatabase/CoulombSum3dDatabase.cpp](Examples/CoulombSum3dDatabase/CoulombSum3dDatabase.cpp). The schema behind these operations is documented in [ERD.md](ERD.md).

## Enable database usage

Enable the database build option in premake5 (use the option `--database`), which defines `KTT_DATABASE` and includes the database header in the build.

## Load results and stats

Open a `Database` (default location `~/.local/share/ktt/ktt.db` on Linux, or `$XDG_DATA_HOME/ktt/ktt.db` when set; `%LOCALAPPDATA%\ktt\ktt.db` on Windows) and use `GetDatabaseTuningInfo` to key queries against a tuning problem:

- `SimpleGetBestResults` returns the fastest stored results for that tuning problem.
- `GetBestResults` takes a `GetBestResultsQuery` with optional device and input-data predicates for fine-grained filtering.
- `GetStatsForSource` returns aggregate counts (spaces, devices, runs, results) for a source fingerprint.

## Save results

Capture results and persist them for later runs with `SaveResults`. You can attach free-form run metadata via `TuningInfo::inputData`, which is later filterable through the input predicate of a `GetBestResultsQuery`.

## Sync from another database

`SyncFromFile` merges all tuning data from another database file into the current one. Runs are matched by their GUID, so
runs that already exist are skipped — the operation is idempotent and safe to repeat. Each newly copied run
brings along its tuning source, tuning space, device and results, and the copy runs inside a single
transaction (a failure leaves the current database unchanged). The other database is opened read-only.

## Output format and JSON indentation

Database stores tuning results as text in the database. The serialization format and, for JSON, its
indentation are controlled per call through the `SaveOptions` argument to `SaveResults` (not the
constructor). Indentation is useful for SQL reader apps (e.g., DBeaver) that display JSON text more
clearly. `SaveOptions` defaults to JSON with an indent of 2; XML is also supported (indentation is ignored for XML).

If you need a custom database path, construct `Database` with a filesystem path; an already-open `sqlite3` connection can also be adopted.
