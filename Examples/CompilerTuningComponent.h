#include "CliComponent.h"
#include <Ktt.h>

class CompilerTuningComponent {
public:
    CompilerTuningComponent(ktt::Tuner &tuner, ktt::KernelId kernel);

    // Public virtual despite NVI guidelines because they would just be thin wrappers otherwise.
    // Follows "Do not generalize prematurely" and "you aren't gonna need it"
    virtual void InitCLIOptions(std::vector<CliOption> &options);
    virtual void AddCompilerParameter(const std::string &name, const std::vector<std::string> &values);
    virtual void Run();

protected:
    ktt::Tuner &m_tuner;
    ktt::KernelId m_kernel;
    bool useSeparateTuning;
};

class NoCompilerTuning : public CompilerTuningComponent {
public:
    using CompilerTuningComponent::CompilerTuningComponent;

    void AddCompilerParameter(const std::string &name, const std::vector<std::string> &values) override;
    void InitCLIOptions(std::vector<CliOption> &options) override;
    void Run() override;
};