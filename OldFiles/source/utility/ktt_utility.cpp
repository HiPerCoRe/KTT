#include <stdexcept>
#include <utility/ktt_utility.h>

namespace ktt
{

uint64_t convertTime(const uint64_t timeInNanoseconds, const TimeUnit targetUnit)
{
    switch (targetUnit)
    {
    case TimeUnit::Nanoseconds:
        return timeInNanoseconds;
    case TimeUnit::Microseconds:
        return timeInNanoseconds / 1'000;
    case TimeUnit::Milliseconds:
        return timeInNanoseconds / 1'000'000;
    case TimeUnit::Seconds:
        return timeInNanoseconds / 1'000'000'000;
    default:
        throw std::runtime_error("Unknown time unit");
    }
}

std::string getTimeUnitTag(const TimeUnit unit)
{
    switch (unit)
    {
    case TimeUnit::Nanoseconds:
        return std::string("ns");
    case TimeUnit::Microseconds:
        return std::string("us");
    case TimeUnit::Milliseconds:
        return std::string("ms");
    case TimeUnit::Seconds:
        return std::string("s");
    default:
        throw std::runtime_error("Unknown time unit");
    }
}

size_t roundUp(const size_t number, const size_t multiple)
{
    if (multiple == 0)
    {
        return number;
    }

    return ((number + multiple - 1) / multiple) * multiple;
}

std::vector<size_t> roundUpGlobalSize(const std::vector<size_t>& globalSize, const std::vector<size_t>& localSize)
{
    std::vector<size_t> result;

    for (size_t i = 0; i < globalSize.size(); i++)
    {
        size_t multiple = roundUp(globalSize.at(i), localSize.at(i));
        result.push_back(multiple);
    }

    return result;
}

} // namespace ktt
