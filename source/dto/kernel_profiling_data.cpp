#include "kernel_profiling_data.h"

namespace ktt
{

KernelProfilingData::KernelProfilingData() :
    achievedOccupancy(-1.0f),
    validFlag(false)
{}

void KernelProfilingData::setAchievedOccupancy(const float achievedOccupancy)
{
    this->achievedOccupancy = achievedOccupancy;
}

void KernelProfilingData::setValid(const bool flag)
{
    validFlag = flag;
}

float KernelProfilingData::getAchievedOccupancy() const
{
    return achievedOccupancy;
}

bool KernelProfilingData::isValid() const
{
    return validFlag;
}

} // namespace ktt
