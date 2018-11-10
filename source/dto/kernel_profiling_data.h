#pragma once

namespace ktt
{

class KernelProfilingData
{
public:
    KernelProfilingData();

    void setAchievedOccupancy(const float achievedOccupancy);
    void setValid(const bool flag);

    float getAchievedOccupancy() const;
    bool isValid() const;

private:
    float achievedOccupancy;
    bool validFlag;
};

} // namespace ktt
