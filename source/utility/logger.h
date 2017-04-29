#pragma once

#include <ostream>
#include <string>

#include "../enum/logging_target.h"

namespace ktt
{

class Logger
{
public:
    Logger();

    void setLoggingTarget(std::ostream& outputTarget);
    void setLoggingTarget(const std::string& filePath);

    void log(const std::string& message) const;

private:
    std::ostream* outputTarget;
    bool filePathValid;
    std::string filePath;
};

} // namespace ktt
