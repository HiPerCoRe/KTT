#include "CliComponent.h"
#include <functional>
#include <vector>
#include <assert.h>
#include <iostream>

using namespace std;

CliOption::CliOption(function<void (const vector<string> &)> callback, const string &trigger, 
                     const string &description, const string &argumentDescriptions,
                     const size_t argumentCount, optional<size_t> minArgumentCount)
        : m_callback(callback), m_trigger(trigger), m_description(description),
          m_argumentDescriptions(argumentDescriptions), m_argumentCount(argumentCount),
          m_minArgumentCount(minArgumentCount.value_or(argumentCount))
{
}

string CliOption::get_string() const
{
    return m_trigger + " " + m_argumentDescriptions + "\n\t" + m_description;
}

void CliOption::PrintArgCountRangeError(string trigger, int low, int high) 
{
    assert(low <= high);
    string range;
    if (low == high) range = to_string(low);
    else range = to_string(low) + "-" + to_string(high);
    cerr << trigger << " expects " + range + " argument(s) to be passed!" << endl;
}

bool CliOption::TryTrigger(int argc, char **argv, int &i) const {
    assert(i < argc);
    if (argv[i] != m_trigger) return false;
    if (i + m_minArgumentCount >= static_cast<size_t>(argc))
    {
        PrintArgCountRangeError(m_trigger, m_minArgumentCount, m_argumentCount);
        exit(1);
    }
    vector<string> arguments;
    ++i;
    while (i < argc && string(argv[i]).find("--") != 0) {
        arguments.push_back(argv[i]);
        ++i;
    }
    if (arguments.size() < m_minArgumentCount || arguments.size() > m_argumentCount)
    {
        PrintArgCountRangeError(m_trigger, m_minArgumentCount, m_argumentCount);
        exit(1);
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