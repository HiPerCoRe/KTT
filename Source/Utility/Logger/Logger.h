#pragma once

#include <ostream>
#include <string>

#include <Utility/Logger/LoggingLevel.h>
#include <Utility/DisableCopyMove.h>

namespace ktt
{

class Logger : public DisableCopyMove
{
public:
    static Logger& GetLogger();

    void SetLoggingLevel(const LoggingLevel level);
    void SetLoggingTarget(std::ostream& target);
    void SetLoggingTarget(const std::string& file);

    LoggingLevel GetLoggingLevel() const;

    void Log(const LoggingLevel level, const std::string& message) const;
    static void LogError(const std::string& message);
    static void LogWarning(const std::string& message);
    static void LogInfo(const std::string& message);
    static void LogDebug(const std::string& message);

private:
    LoggingLevel m_Level;
    std::ostream* m_OutputTarget;
    std::string m_File;
    bool m_FileValid;

    Logger();
    static std::string GetLoggingLevelString(const LoggingLevel level);
};

} // namespace ktt
