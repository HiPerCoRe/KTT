#pragma once

#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <TuningLoaderPlatform.h>

namespace ktt
{

class TunerCommand;
class TunerContext;

class KTT_LOADER_API TuningLoader
{
public:
    TuningLoader();
    ~TuningLoader();

    void LoadTuningFile(const std::string& file, const std::map<std::string, std::string>& parameters = {});
    void ExecuteCommands();

private:
    std::unique_ptr<TunerContext> m_Context;
    std::vector<std::unique_ptr<TunerCommand>> m_Commands;

    void DeserializeCommands(const std::string& tuningScript);
    static bool ValidateFormat(const std::string& tuningScript);
    static std::string InjectParameters(const std::string& file, const std::map<std::string, std::string>& parameters);
};

} // namespace ktt
