#pragma once

#include <string>
#include <functional>
#include <vector>
#include <assert.h>

/** @class A value-class used for individual CLI options. Contains the trigger, description for --help, and the function to call when triggered. */
class CliOption
{
    std::function<void (const std::vector<std::string> &)> m_callback;
    const std::string m_trigger;
    const std::string m_description;
    const std::string m_argumentDescriptions;
    const int m_argumentCount;

public:
    /** @fn Constructor.
      * @param callback The function to call when triggered. Usually a lambda with captured variables, or captured `this`.
      * @param trigger The string that will be matched with user input arguments when TryTrigger is called in CliComponent::ProcessInput.
      * @param description The description that will be displayed under the trigger in --help.
      * @param argumentDescriptions Default "". Argument descriptions that will be displayed next to the trigger in --help.
      * @param argumentCount Default 0. Number of arguments that will be passed to the callback after the option is triggered. It is strict: the user has to pass exactly this amount.
      */
    CliOption(std::function<void (const std::vector<std::string> &)> callback, const std::string &trigger, const std::string &description,
              const std::string &argumentDescriptions = "", const int argumentCount = 0);

    /** @fn Get a text representation of the option. Used by --help's callback */
    std::string get_string() const;

    /** @fn Tries to match the i-th argument to the trigger. If matched, consumes argumentCount arguments and calls the callback.
      * @param argc Number of arguments passed from main
      * @param argv Arguments passed from main
      * @param i Index of currently processed argument. Passed as a reference.*/
    bool TryTrigger(int argc, char **argv, int &i) const;
};

/** @class Component managing the CLI system. Lets one add options and processes input. */
class CliComponent {
public:
    /** @fn Constructor. Initializes the --help option. */
    CliComponent();

    /** @fn Adds an option to the CLI. Is automatically included in the --help option.
      * @param option The option to be added, can be constructed in-place (`AddOption({...})`). See CliOption for parameters.
      */
    void AddOption(const CliOption &option);

    /** @fn Processes command line input.
      * @param argc Number of arguments from the main function.
      * @param argv The arguments themselves from the main function.*/
    void ProcessInput(int argc, char **argv);

protected:
    std::vector<CliOption> m_options;
};