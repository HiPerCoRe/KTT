#pragma once

#include <ostream>
#include <string>

#include <Utility/Logger/LoggingLevel.h>

namespace ktt
{

class Logger
{
public:
    static Logger& GetLogger();

    void SetLoggingLevel(const LoggingLevel level);
    void SetLoggingTarget(std::ostream& target);
    void SetLoggingTarget(const std::string& file);

    void Log(const LoggingLevel level, const std::string& message) const;
    static void LogError(const std::string& message);
    static void LogWarning(const std::string& message);
    static void LogInfo(const std::string& message);
    static void LogDebug(const std::string& message);

    Logger(const Logger&) = delete;
    Logger(Logger&&) = delete;
    void operator=(const Logger&) = delete;
    void operator=(Logger&&) = delete;

private:
    LoggingLevel m_Level;
    std::ostream* m_OutputTarget;
    std::string m_File;
    bool m_FileValid;

    Logger();
    static std::string GetLoggingLevelString(const LoggingLevel level);
};

} // namespace ktt
