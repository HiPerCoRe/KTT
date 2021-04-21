#include <Utility/StringUtility.h>

namespace ktt
{

bool StartsWith(const std::string& target, const std::string& prefix)
{
    return target.rfind(prefix, 0) == 0;
}

void RemoveTrailingZero(std::string& target)
{
    if (target.empty())
    {
        return;
    }

    if (target.back() == '\0')
    {
        target.pop_back();
    }
}

} // namespace ktt
