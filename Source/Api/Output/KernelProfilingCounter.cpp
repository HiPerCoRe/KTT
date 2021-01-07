#include <Api/Output/KernelProfilingCounter.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

KernelProfilingCounter::KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const int64_t value) :
    m_Name(name),
    m_Type(type),
    m_Value(value)
{}

KernelProfilingCounter::KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const uint64_t value) :
    m_Name(name),
    m_Type(type),
    m_Value(value)
{}

KernelProfilingCounter::KernelProfilingCounter(const std::string& name, const ProfilingCounterType type, const double value) :
    m_Name(name),
    m_Type(type),
    m_Value(value)
{}

const std::string& KernelProfilingCounter::GetName() const
{
    return m_Name;
}

ProfilingCounterType KernelProfilingCounter::GetType() const
{
    return m_Type;
}

int64_t KernelProfilingCounter::GetValueInt() const
{
    if (m_Type != ProfilingCounterType::Int)
    {
        throw KttException("Attempting to retrieve value of a kernel profiling counter with incorrect data type");
    }

    return std::get<int64_t>(m_Value);
}

uint64_t KernelProfilingCounter::GetValueUint() const
{
    if (m_Type != ProfilingCounterType::UnsignedInt && m_Type != ProfilingCounterType::Throughput
        && m_Type != ProfilingCounterType::UtilizationLevel)
    {
        throw KttException("Attempting to retrieve value of a kernel profiling counter with incorrect data type");
    }

    return std::get<uint64_t>(m_Value);
}

double KernelProfilingCounter::GetValueDouble() const
{
    if (m_Type != ProfilingCounterType::Double && m_Type != ProfilingCounterType::Percent)
    {
        throw KttException("Attempting to retrieve value of a kernel profiling counter with incorrect data type");
    }

    return std::get<double>(m_Value);
}

bool KernelProfilingCounter::operator==(const KernelProfilingCounter& other) const
{
    return m_Name == other.m_Name;
}

bool KernelProfilingCounter::operator!=(const KernelProfilingCounter& other) const
{
    return !(*this == other);
}

bool KernelProfilingCounter::operator<(const KernelProfilingCounter& other) const
{
    return m_Name < other.m_Name;
}

} // namespace ktt
