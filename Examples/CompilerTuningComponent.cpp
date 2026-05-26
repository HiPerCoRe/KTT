#include "CompilerTuningComponent.h"

CompilerTuningComponent::CompilerTuningComponent(ktt::Tuner &tuner, ktt::KernelId kernel)
    : m_tuner(tuner), m_kernel(kernel) {
    useSeparateTuning = false;
}

void CompilerTuningComponent::AddCompilerParameter(const std::string &name, const std::vector<std::string> &values) {
    if (useSeparateTuning) {

    }
}

void CompilerTuningComponent::InitCLIOptions(std::vector<CliOption> &options) {
    // TODO: Implement CLI options initialization
}

void CompilerTuningComponent::Run() {
    // TODO: Implement main execution logic
}


void NoCompilerTuning::AddCompilerParameter(const std::string &name, const std::vector<std::string> &values) {
}

void NoCompilerTuning::InitCLIOptions(std::vector<CliOption> &options) {
}

void NoCompilerTuning::Run() {
}
