#pragma once

#include <ostream>
#include <string>
#include "enum/logging_level.h"

namespace ktt
{

class Logger
{
public:
    Logger();

    void setLoggingLevel(const LoggingLevel level);
    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);

    void log(const LoggingLevel level, const std::string& message) const;

private:
    LoggingLevel level;
    std::ostream* outputTarget;
    bool filePathValid;
    std::string filePath;

    static std::string getLoggingLevelString(const LoggingLevel level);
};

} // namespace ktt
