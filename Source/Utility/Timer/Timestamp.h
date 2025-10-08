#pragma once

#include <string>
#include <sstream>
#include <chrono>
#include <date.h>

namespace ktt
{

class Timestamp 
{
public:
    static inline const std::string GetTimestamp()
    {
        using namespace date;
        const auto now = std::chrono::system_clock::now();
        const auto today = date::floor<days>(now);
        std::stringstream stream;
        stream << today << ' ' << make_time(now - today) << " UTC";
        return stream.str();
    }
};

} //ktt namespace
