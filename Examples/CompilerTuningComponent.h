#include "CliComponent.h"
#include "Tuner.h"
#include <memory>

/** @class A component taking care of compiler parameter tuning. Split off ExampleBase due to being an optional feature. */
class CompilerTuningComponent {
public:
    /** @fn Constructor.
      * @param tuner Reference to the tuner shared pointer. Can be null now, but must be valid by the time AddCompilerParameter is being called.
      * @param kernel Reference to the tuned kernel id.
      */
    CompilerTuningComponent(std::shared_ptr<ktt::Tuner> &tuner, ktt::KernelId &kernel);

    // Public virtual despite NVI guidelines because they would just be thin wrappers otherwise.
    // Follows "Do not generalize prematurely" and "you aren't gonna need it"

    /** @fn Adds its own CLI option to the CliComponent.
      * @param cli The CliComponent to add the option to.
      */
    virtual void InitCLIOptions(CliComponent &cli);

    /** @fn Adds a compiler parameter. Internally calls either m_tuner->AddCompilerParameter(...) or m_tuner->AddSeparateCompilerParameter(...)
      * depending on useSeparateTuning value.
      * @param name The parameter name will be passed as an option to the compiler
      * @param values Different values that can be inserted after the parameter/option. If empty the tuner will either insert
      * or not insert the whole parameter/option, treating it like a binary flag.
      */
    virtual void AddCompilerParameter(const std::string &name, const std::vector<std::string> &values = {});

    /** @fn Runs the compiler tuning process. Does nothing if separate tuning is not enabled. */
    virtual void Run();

protected:
    std::shared_ptr<ktt::Tuner> &m_tuner;
    ktt::KernelId &m_kernel;
    bool useSeparateTuning;
};

/** @class A null version of CompilerTuningComponent. Lets one have the component disabled using polymorphism. */
class NoCompilerTuning : public CompilerTuningComponent {
public:
    using CompilerTuningComponent::CompilerTuningComponent;

    void AddCompilerParameter(const std::string &name, const std::vector<std::string> &values = {}) override;
    void InitCLIOptions(CliComponent &cli) override;
    void Run() override;
};