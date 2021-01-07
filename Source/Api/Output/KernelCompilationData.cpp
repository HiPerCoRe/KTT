#include <Api/Output/KernelCompilationData.h>

namespace ktt
{

KernelCompilationData::KernelCompilationData() :
    m_MaxWorkGroupSize(0),
    m_LocalMemorySize(0),
    m_PrivateMemorySize(0),
    m_ConstantMemorySize(0),
    m_RegistersCount(0)
{}

} // namespace ktt
