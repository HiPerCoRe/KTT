#pragma once

#include <KttTypes.h>

namespace ktt
{

class TransferResult
{
public:
    TransferResult();
    explicit TransferResult(const Nanoseconds duration, const Nanoseconds overhead);

    void SetDuration(const Nanoseconds duration);
    void SetOverhead(const Nanoseconds overhead);

    Nanoseconds GetDuration() const;
    Nanoseconds GetOverhead() const;

private:
    Nanoseconds m_Duration;
    Nanoseconds m_Overhead;
};

} // namespace ktt
