#pragma once

#include <cstddef>
#include <map>
#include <string>

namespace ktt
{

struct CUPTIMetric
{
public:
    CUPTIMetric() :
        name(""),
        rangeCount(0)
    {}

    std::string getHWUnit() const
    {
        return name.substr(0, name.find("__"));
    }

    std::string name;
    size_t rangeCount;
    std::map<std::string, double> rangeToMetricValue;
};

} // namespace ktt
