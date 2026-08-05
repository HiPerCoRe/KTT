#include "CliComponent.h"
#include <functional>
#include <vector>
#include <assert.h>
#include <iostream>

using namespace std;

CliOption::CliOption(function<void (const vector<string> &)> callback, const string &trigger, const string &description,
              const string &argumentDescriptions, const int argumentCount)
        : m_callback(callback), m_trigger(trigger), m_description(description),
          m_argumentDescriptions(argumentDescriptions), m_argumentCount(argumentCount)
{
}

string CliOption::get_string() const
{
    return m_trigger + " " + m_argumentDescriptions + "\n\t" + m_description;
}

bool CliOption::TryTrigger(int argc, char **argv, int &i) const {
    assert(i < argc);
    if (argv[i] != m_trigger) return false;
    if (i + m_argumentCount >= argc)
    {
        cerr << m_trigger << " expects a value to be passed!" << endl;
        exit(1);
    }
    vector<string> arguments;
    for (int j = 0; j < m_argumentCount; ++j) {
        arguments.push_back(argv[++i]);
    }
    m_callback(arguments);
    return true;
}

CliComponent::CliComponent()
{
    AddOption({[this](const vector<string> &) {
        cout << "Usage: program [options]" << endl << endl;
        cout << "Options:" << endl;
        for (const auto& option : m_options) {
            cout << option.get_string() << endl;
        }
        exit(0);
    }, "--help", "Show this help message and exit."});
}

void CliComponent::AddOption(const CliOption &cliOption) {
    m_options.push_back(cliOption);
}

void CliComponent::ProcessInput(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        bool triggered = false;
        for (const auto& option : m_options) {
            if (option.TryTrigger(argc, argv, i)) {
                triggered = true;
                break;
            }
        }
        if (!triggered) {
            cerr << argv[i] << " is not a valid option.\n";
            exit(1);
        }
    }
}