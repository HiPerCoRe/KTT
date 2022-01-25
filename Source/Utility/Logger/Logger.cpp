#include <fstream>
#include <iostream>

#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

Logger& Logger::GetLogger()
{
    static Logger instance;
    return instance;
}

void Logger::SetLoggingLevel(const LoggingLevel level)
{
    m_Level = level;
}

void Logger::SetLoggingTarget(std::ostream& target)
{
    m_OutputTarget = &target;
    m_FileValid = false;
}

void Logger::SetLoggingTarget(const std::string& file)
{
    m_File = file;
    m_FileValid = true;
}

void Logger::Log(const LoggingLevel level, const std::string& message) const
{
    if (static_cast<int>(m_Level) < static_cast<int>(level))
    {
        return;
    }

    if (m_FileValid)
    {
        std::ofstream outputFile(m_File, std::ios::app | std::ios_base::out);

        if (!outputFile.is_open())
        {
            std::cerr << "Unable to open file: " << m_File << std::endl;
            return;
        }

        outputFile << GetLoggingLevelString(level) << " " << message << std::endl;
    }
    else
    {
        *m_OutputTarget << GetLoggingLevelString(level) << " " << message << std::endl;
    }
}

void Logger::LogError(const std::string& message)
{
    GetLogger().Log(LoggingLevel::Error, message);
}

void Logger::LogWarning(const std::string& message)
{
    GetLogger().Log(LoggingLevel::Warning, message);
}

void Logger::LogInfo(const std::string& message)
{
    GetLogger().Log(LoggingLevel::Info, message);
}

void Logger::LogDebug(const std::string& message)
{
    GetLogger().Log(LoggingLevel::Debug, message);
}

Logger::Logger() :
    m_Level(LoggingLevel::Info),
    m_OutputTarget(&std::clog),
    m_FileValid(false)
{}

std::string Logger::GetLoggingLevelString(const LoggingLevel level)
{
    switch (level)
    {
    case LoggingLevel::Off:
        return "[Off]";
    case LoggingLevel::Error:
        return "[Error]";
    case LoggingLevel::Warning:
        return "[Warning]";
    case LoggingLevel::Info:
        return "[Info]";
    case LoggingLevel::Debug:
        return "[Debug]";
    default:
        KttError("Unhandled logging level value");
        return "";
    }
}

} // namespace ktt
