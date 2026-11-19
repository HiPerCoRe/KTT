#include "../Common/CoulombSum3dDatabaseExample.h"
#include <Database/Remote/RemoteDatabase.h>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace std;

/** @class Tunes the 3D Coulomb sum and stores the results on a ktt-db server through ktt::db::RemoteDatabase.
  * Build with: premake5 gmake --remote-database [--remote-database-host=https://...]
  * The server is fixed at build time (RemoteDatabase::GetHost, default https://supercuda.fi.muni.cz:8443).
  * The API key (created in the web app's settings) is read from the KTT_DB_API_KEY environment variable when it is set,
  * otherwise from the key file (RemoteDatabase::GetDefaultApiKeyPath, ~/.local/share/ktt/api-key on Linux).
  * See Examples/Database/CoulombSum3dLocal for the same example against the local SQLite database.
  */
class CoulombSum3dRemote : public CoulombSum3dDatabaseExample {
protected:
    CoulombSum3dRemote(int argc, char **argv, string exampleFolderPath, string defaultKernelFileBaseName) :
        CoulombSum3dDatabaseExample(argc, argv, exampleFolderPath, defaultKernelFileBaseName)
    {}

public:
    // Same flow as CoulombSum3dLocal::Run(): load the best known results, re-run the best one, tune, and save the
    // new results, but against the ktt-db server. Every remote call throws KttException on failure (network error,
    // wrong key, rejected request), which main() reports.
    void Run()
    {
        /**
         * --- Remote database: connect ---
         */
        ktt::db::RemoteDatabaseOptions options;
        // The local development backend has a self-signed certificate: pass its CA with --ca-bundle,
        // or (for testing only) skip verification with --insecure.
        options.verifyTls = !m_insecure;
        if (!m_caBundle.empty())
        {
            options.caBundle = m_caBundle;
        }
        // After a save the server reports whether it has the kernel's source code; if not, the client uploads it
        // through this provider so the web app can show it next to the results.
        options.sourceCodeProvider = ktt::db::MakeSourceCodeProvider(*m_tuner, {m_kernel});

        // Construction does not contact the server, so a wrong key or host shows up on the first call.
        const char *apiKey = std::getenv("KTT_DB_API_KEY");
        const ktt::db::RemoteDatabase db = (apiKey != nullptr && *apiKey != '\0')
            ? ktt::db::RemoteDatabase(apiKey, std::move(options))
            : ktt::db::RemoteDatabase(std::move(options));

        cout << "Using ktt-db server " << ktt::db::RemoteDatabase::GetHost() << endl;

        const auto tuningInfo = m_tuner->GetDatabaseTuningInfo(m_kernel);
        const string inputData = GetInputData();

        /**
         * --- Remote database: what does the server know about this kernel? ---
         */
        // Counts cover only the runs the API key can see.
        if (const auto stats = db.GetStatsForSource(tuningInfo.spaceInfo.sourceFingerprint))
        {
            cout << "Server has " << stats->runCount << " run(s) with " << stats->resultCount
                 << " result(s) of this kernel on " << stats->deviceCount << " device(s)." << endl;
        }
        else
        {
            cout << "Server has no runs of this kernel yet." << endl;
        }

        /**
         * --- Remote database: load previously stored best results ---
         */
        // Same tuning space on a matching device (same model; for CUDA, a close compute capability).
        const auto simpleResults = db.SimpleGetBestResults(tuningInfo, 50);

        // Predicate variant: same tuning space, on any device, but only runs with the same input data.
        // The predicates run on the client, so this pages through the space's runs in several requests.
        const ktt::db::GetBestResultsQuery query{
            tuningInfo.spaceInfo,
            std::nullopt, // no device filter
            std::function<bool(const string &)>([&inputData](const string &runInput) {
                return runInput == inputData;
            }),
            50
        };
        const auto queryResults = db.GetBestResults(query);

        cout << "Loaded " << simpleResults.size() << " best result(s) for this device "
             << "(" << queryResults.size() << " across all devices with input '" << inputData << "')." << endl;

        /**
         * --- Remote database: server-side query (remote only) ---
         */
        // The same kind of selection as the predicate query above, written as a filter and run on the server in one
        // request. Fingerprints are string literals in the filter language.
        ktt::db::ResultQuery resultQuery;
        resultQuery.filter = "source/fingerprint eq '" + to_string(tuningInfo.spaceInfo.sourceFingerprint) + "'"
                           + " and run/input_data eq '" + inputData + "'";
        resultQuery.orderBy = "duration asc";
        resultQuery.top = 5;
        resultQuery.includeResult = false; // the configuration and durations are enough here

        const auto items = db.QueryResults(resultQuery);
        cout << "Top " << items.size() << " result(s) for this input on the server:" << endl;
        for (const auto &item : items)
        {
            cout << "  " << item.duration << " ns on " << item.device.name << " (run " << item.runId << "):";
            for (const auto &[name, value] : item.configuration)
            {
                cout << " " << name << "=";
                std::visit([](const auto &v) { cout << v; }, value);
            }
            cout << endl;
        }

        /**
         * --- Remote database: run the best previously known configuration ---
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
         * --- Remote database: save this run's results ---
         */
        // Needs a key with write access. The run is linked to every realm of the key. The names are applied only
        // when the source or space first appears in a realm.
        auto save = m_tuner->GetDatabaseTuningInfo(m_kernel);
        save.inputData = inputData;
        save.spaceInfo.sourceName = "CoulombSum3d";
        // The tuning space differs per compute API (see InitTuningSpace), so the space name says which one it is.
        save.spaceInfo.spaceName = m_computeApi == ktt::ComputeApi::CUDA ? "CoulombSum3d CUDA"
                                 : m_computeApi == ktt::ComputeApi::OpenCL ? "CoulombSum3d OpenCL"
                                 : "CoulombSum3d C++";
        db.SaveResults(save, results);
        cout << "Saved " << results.size() << " result(s) to the server." << endl;

        // --- Optional: tune compiler options on top of the best kernel config (--sepCompTuning) ---
        m_compilerTuning->Run();
    }

protected:
    friend ExampleBase;

    // CA bundle (PEM) used to verify the server's certificate; set via --ca-bundle.
    string m_caBundle;

    // Skips TLS verification; set via --insecure. For a local test server only.
    bool m_insecure = false;

    void InitCLI() override
    {
        CoulombSum3dDatabaseExample::InitCLI();
        m_cli.AddOption({[this](const vector<string> &args) {
            m_caBundle = args[0];
        }, "--ca-bundle", "CA bundle (PEM) used to verify the ktt-db server's certificate", "<path>", 1});
        m_cli.AddOption({[this](const vector<string> &) {
            m_insecure = true;
        }, "--insecure", "Do not verify the ktt-db server's TLS certificate (testing only)"});
    }
};

int main(int argc, char **argv)
{
    try
    {
        unique_ptr<CoulombSum3dRemote> example = CoulombSum3dRemote::Create<CoulombSum3dRemote>(argc, argv,
            CoulombSum3dDatabaseExample::KernelFolder, CoulombSum3dDatabaseExample::KernelBaseName);
        example->Run();
    }
    catch (const ktt::KttException &exception)
    {
        cerr << "CoulombSum3dRemote failed: " << exception.what() << endl;
        return 1;
    }

    return 0;
}
