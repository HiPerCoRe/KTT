#include "ktt_utility.h"

namespace ktt
{

std::vector<size_t> convertDimensionVector(const DimensionVector& vector)
{
    std::vector<size_t> result;

    result.push_back(std::get<0>(vector));
    result.push_back(std::get<1>(vector));
    result.push_back(std::get<2>(vector));

    return result;
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
