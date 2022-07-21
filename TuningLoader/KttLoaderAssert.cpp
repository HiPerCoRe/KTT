#include <cstdlib>
#include <iostream>

#include <KttLoaderAssert.h>

namespace ktt
{

void LoaderAssertInternal(const char* expressionString, const bool expression, const char* file, const int line,
    const std::string& message)
{
    if (expression)
    {
        return;
    }

    std::cerr << "Assertion failed: " << message << std::endl;
    std::cerr << "Condition: " << expressionString << std::endl;
    std::cerr << "Source location: " << file << ", line: " << line << std::endl;

    std::abort();
}

} // namespace ktt
