#pragma once

#include <cstdint>
#include <string>

#include <Output/TimeConfiguration/TimeUnit.h>
#include <Utility/DisableCopyMove.h>
#include <KttTypes.h>

namespace ktt
{

class TimeConfiguration : public DisableCopyMove
{
public:
    static TimeConfiguration& GetInstance();

    void SetTimeUnit(const TimeUnit unit);

    TimeUnit GetTimeUnit() const;
    std::string GetUnitTag() const;

    uint64_t ConvertFromNanoseconds(const Nanoseconds duration) const;
    double ConvertFromNanosecondsDouble(const Nanoseconds duration) const;

    Nanoseconds ConvertToNanoseconds(const uint64_t duration) const;
    Nanoseconds ConvertToNanosecondsDouble(const double duration) const;

private:
    TimeUnit m_TimeUnit;

    TimeConfiguration();
};

} // namespace ktt
