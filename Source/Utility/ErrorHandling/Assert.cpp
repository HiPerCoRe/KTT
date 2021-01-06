#include <cstdlib>
#include <string>

#include <Utility/Logger/Logger.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

void AssertInternal(const char* expressionString, const bool expression, const char* file, const int line, const char* message)
{
    if (expression)
    {
        return;
    }

    Logger::LogError(std::string("Assertion failed: ") + message);
    Logger::LogError(std::string("Condition: ") + expressionString);
    Logger::LogError(std::string("Source location: ") + file + ", line " + std::to_string(line));

    std::abort();
}

} // namespace ktt
