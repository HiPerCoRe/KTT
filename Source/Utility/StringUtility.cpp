#include <Utility/StringUtility.h>

namespace ktt
{

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
