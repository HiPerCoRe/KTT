#include "../Common/CoulombSum3dDatabaseExample.h"
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace std;

/** @class Tunes the 3D Coulomb sum and stores the results in the local (SQLite) KTT database.
  * Build with: premake5 gmake --database
  * See Examples/Database/CoulombSum3dRemote for the same example against a ktt-db server.
  */
class CoulombSum3dLocal : public CoulombSum3dDatabaseExample {
protected:
    CoulombSum3dLocal(int argc, char **argv, string exampleFolderPath, string defaultKernelFileBaseName) :
        CoulombSum3dDatabaseExample(argc, argv, exampleFolderPath, defaultKernelFileBaseName)
    {}

public:
    // Custom offline run that keeps the tuning results in hand so they can be
    // pushed to / loaded from the KTT database directly. Structurally this mirrors
    // ExampleBase::RunOffline() (offline tuning + optional separate compiler
    // tuning), with the database interaction woven around the Tune() call.
    void Run()
    {
        /**
         * --- Database integration: prep ---
         */
        // Open the database at the path passed via --db, or the default location when omitted.
        const ktt::db::Database db = m_dbPath.empty()
            ? ktt::db::Database()
            : ktt::db::Database(m_dbPath);

        // Sync the database file passed via --db-sync into the opened database.
        // Skipped when --db-sync was not provided.
        if (!m_dbSyncPath.empty())
        {
            db.SyncFrom(ktt::db::Database(m_dbSyncPath));
        }

        const auto tuningInfo = m_tuner->GetDatabaseTuningInfo(m_kernel);

        /**
         * --- Database integration: load previously stored best results ---
         */
        const auto simpleResults = db.SimpleGetBestResults(tuningInfo, 50);

        // Advanced query variant: same tuning space, restricted to CUDA devices.
        const ktt::db::GetBestResultsQuery query{
            tuningInfo.spaceInfo,
            std::function<bool(const ktt::db::DeviceInfo &)>([](const ktt::db::DeviceInfo &device) {
                return device.computeApi == ktt::ComputeApi::CUDA;
            }),
            std::nullopt, // no input filter
            50
        };
        const auto queryResults = db.GetBestResults(query);

        cout << "Loaded " << simpleResults.size() << " best result(s) from the database "
             << "(" << queryResults.size() << " via CUDA-filtered query)." << endl;

        /**
         * --- Database integration: run the best previously known configuration ---
         */
        if (!simpleResults.empty())
        {
            const auto bestConfig = simpleResults[0].GetConfiguration();
            const auto bestResult = m_tuner->Run(m_kernel, bestConfig, {});
            cout << "Re-ran best known configuration: " << bestResult.GetTotalDuration()
                 << " ns" << endl;
        }

        // --- Offline tuning (driven here so we hold the results directly) ---
        const auto results = m_tuner->Tune(m_kernel, std::move(m_stopCondition), m_preciseParams);

        m_tuner->SaveResults(results, "Output", ktt::OutputFormat::JSON);

        /**
         * --- Database integration: save this run's results ---
         */
        auto save = m_tuner->GetDatabaseTuningInfo(m_kernel);
        save.inputData = GetInputData();
        db.SaveResults(save, results, {ktt::OutputFormat::JSON, 2});

        // --- Optional: tune compiler options on top of the best kernel config (--sepCompTuning) ---
        m_compilerTuning->Run();
    }

protected:
    friend ExampleBase;

    // Path of the KTT database to open. Empty by default, in which case the
    // default database location is used; set via --db.
    string m_dbPath;

    // Database file whose runs are synced into the opened database before tuning.
    // Empty by default; set via --db-sync, in which case that file is synced.
    string m_dbSyncPath;

    void InitCLI() override
    {
        CoulombSum3dDatabaseExample::InitCLI();
        m_cli.AddOption({[this](const vector<string> &args) {
            m_dbPath = args[0];
        }, "--db", "Path of the KTT database to open (defaults to the standard location)",
        "<path>", 1});
        m_cli.AddOption({[this](const vector<string> &args) {
            m_dbSyncPath = args[0];
        }, "--db-sync", "Path to a KTT database file to sync into the opened database before tuning",
        "<path>", 1});
    }
};

int main(int argc, char **argv)
{
    unique_ptr<CoulombSum3dLocal> example = CoulombSum3dLocal::Create<CoulombSum3dLocal>(argc, argv,
        CoulombSum3dDatabaseExample::KernelFolder, CoulombSum3dDatabaseExample::KernelBaseName);
    example->Run();

    return 0;
}
