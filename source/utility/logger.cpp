#include <fstream>
#include <iostream>

#include "logger.h"

namespace ktt
{

Logger::Logger() :
    loggingTarget(LoggingTarget::Clog),
    filePath(std::string(""))
{}

void Logger::setLoggingTarget(const LoggingTarget& loggingTarget)
{
    this->loggingTarget = loggingTarget;
}

void Logger::setFilePath(const std::string& filePath)
{
    this->filePath = filePath;
}

void Logger::log(const std::string& message) const
{
    if (loggingTarget == LoggingTarget::Cout)
    {
        std::cout << message << std::endl;
    }
    else if (loggingTarget == LoggingTarget::Cerr)
    {
        std::cerr << message << std::endl;
    }
    else if (loggingTarget == LoggingTarget::Clog)
    {
        std::clog << message << std::endl;
    }
    else if (loggingTarget == LoggingTarget::File)
    {
        std::ofstream outputFile(filePath, std::ios::app | std::ios_base::out);

        if (!outputFile.is_open())
        {
            std::cerr << "Unable to open file: " << filePath << std::endl;
        }
        outputFile << message << std::endl;
    }
}

} // namespace ktt
