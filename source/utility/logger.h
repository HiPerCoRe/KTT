#pragma once

#include <ostream>
#include <string>
#include <enum/logging_level.h>

namespace ktt
{

class Logger
{
public:
    static Logger& getLogger();

    void setLoggingLevel(const LoggingLevel level);
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);
    void log(const LoggingLevel level, const std::string& message) const;

    static void logError(const std::string& message);
    static void logWarning(const std::string& message);
    static void logInfo(const std::string& message);
    static void logDebug(const std::string& message);

    Logger(const Logger&) = delete;
    Logger(Logger&&) = delete;
    void operator=(const Logger&) = delete;
    void operator=(Logger&&) = delete;

private:
    LoggingLevel level;
    std::ostream* outputTarget;
    bool filePathValid;
    std::string filePath;

    Logger();
    static std::string getLoggingLevelString(const LoggingLevel level);
};

} // namespace ktt
