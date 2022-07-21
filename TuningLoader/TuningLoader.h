#pragma once

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

    void LoadTuningFile(const std::string& file);
    void ExecuteCommands();

private:
    std::unique_ptr<TunerContext> m_Context;
    std::vector<std::unique_ptr<TunerCommand>> m_Commands;
};

} // namespace ktt
