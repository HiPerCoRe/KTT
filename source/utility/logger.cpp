#include <fstream>
#include <iostream>
#include "logger.h"

namespace ktt
{

Logger::Logger() :
    outputTarget(&std::clog),
    filePathValid(false)
{}

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

void Logger::log(const std::string& message) const
{
    if (filePathValid)
    {
        std::ofstream outputFile(filePath, std::ios::app | std::ios_base::out);

        if (!outputFile.is_open())
        {
            std::cerr << "Unable to open file: " << filePath << std::endl;
        }
        outputFile << message << std::endl;
    }
    else
    {
        *outputTarget << message << std::endl;
    }
}

} // namespace ktt
