#pragma once

#include <string>
#include <functional>
#include <vector>
#include <assert.h>

class CliOption
{
    std::function<void (const std::vector<std::string> &)> m_callback;
    const std::string m_trigger;
    const std::string m_description;
    const std::string m_argumentDescriptions;
    const int m_argumentCount;

public:
    CliOption(std::function<void (const std::vector<std::string> &)> callback, const std::string &trigger, const std::string &description,
              const std::string &argumentDescriptions = "", const int argumentCount = 0);

    std::string get_string() const;

    bool TryTrigger(int argc, char **argv, int &i) const;
};

class CliComponent {
public:
    CliComponent();
    void AddOption(const CliOption &option);
    void ProcessInput(int argc, char **argv);

protected:
    std::vector<CliOption> m_options;
};