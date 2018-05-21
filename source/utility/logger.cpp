#include <fstream>
#include <iostream>
#include <stdexcept>
#include "logger.h"

namespace ktt
{

Logger::Logger() :
    level(LoggingLevel::Info),
    outputTarget(&std::clog),
    filePathValid(false)
{}

void Logger::setLoggingLevel(const LoggingLevel level)
{
    this->level = level;
}

void Logger::setLoggingTarget(std::ostream& outputTarget)
{
    this->outputTarget = &outputTarget;
    filePathValid = false;
}

void Logger::setLoggingTarget(const std::string& filePath)
{
    this->filePath = filePath;
    filePathValid = true;
}

void Logger::log(const LoggingLevel level, const std::string& message) const
{
    if (static_cast<int>(this->level) < static_cast<int>(level))
    {
        return;
    }

    if (filePathValid)
    {
        std::ofstream outputFile(filePath, std::ios::app | std::ios_base::out);

        if (!outputFile.is_open())
        {
            std::cerr << "Unable to open file: " << filePath << std::endl;
            return;
        }
        outputFile << getLoggingLevelString(level) << " " << message << std::endl;
    }
    else
    {
        *outputTarget << getLoggingLevelString(level) << " " << message << std::endl;
    }
}

std::string Logger::getLoggingLevelString(const LoggingLevel level)
{
    switch (level)
    {
    case LoggingLevel::Off:
        return std::string("[OFF]");
    case LoggingLevel::Error:
        return std::string("[ERROR]");
    case LoggingLevel::Warning:
        return std::string("[WARNING]");
    case LoggingLevel::Info:
        return std::string("[INFO]");
    case LoggingLevel::Debug:
        return std::string("[DEBUG]");
    default:
        throw std::runtime_error("Unknown logging level");
    }
}

} // namespace ktt
