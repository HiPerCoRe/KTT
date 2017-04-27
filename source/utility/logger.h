#pragma once

#include <string>

#include "../enum/logging_target.h"

namespace ktt
{

class Logger
{
public:
    Logger();

    void setLoggingTarget(const LoggingTarget& loggingTarget);
    void setFilePath(const std::string& filePath);

    void log(const std::string& message) const;

private:
    LoggingTarget loggingTarget;
    std::string filePath;
};

} // namespace ktt
