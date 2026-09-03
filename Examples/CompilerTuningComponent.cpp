#include "CompilerTuningComponent.h"
#include "CliComponent.h"
#include "RunStats.hpp"
#include <cassert>

using namespace std;

CompilerTuningComponent::CompilerTuningComponent(std::shared_ptr<ktt::Tuner> &tuner, ktt::KernelId &kernel):
    m_tuner(tuner),
    m_kernel(kernel),
    useSeparateTuning(false)
{}

void CompilerTuningComponent::AddCompilerParameter(const string &name, const vector<string> &values)
{
    if (useSeparateTuning) m_tuner->AddSeparateCompilerParameter(m_kernel, name, values);
    else m_tuner->AddCompilerParameter(m_kernel, name, values);
}

void CompilerTuningComponent::InitCLIOptions(CliComponent &cli) {
    cli.AddOption({[this](const vector<string> &) {
        useSeparateTuning = true;
    }, "--sepCompTuning", "Enable separate compiler parameter tuning."});
}

void CompilerTuningComponent::Run() {
    if (!useSeparateTuning) return;
    auto bestConfig = m_tuner->GetBestConfiguration(m_kernel);
    if (!bestConfig.IsValid()) {
        cout << "No valid configuration was found. Skipping separate compiler parameter tuning.\n";
        return;
    }
    std::cout << "\nTuning compiler options on top of best kernel configuration..." << std::endl;
    const auto optResults = m_tuner->TuneOptions(m_kernel, bestConfig);
    m_tuner->SaveResults(optResults, "CoulombSumOptionsOutput", ktt::OutputFormat::JSON);

    RunStats stats;

    for (const auto& optResult : optResults)
    {
        stats.Update(optResult);
    }

    stats.Print("Option tuning stats");
}

void NoCompilerTuning::AddCompilerParameter(const string &, const vector<string> &) {
    assert(false && "Example needs to call UseCompilerTuning() in the constructor to use this feature.");
}

void NoCompilerTuning::InitCLIOptions(CliComponent &) {
}

void NoCompilerTuning::Run() {
}
