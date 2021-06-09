#include <KernelRunner/ValidationData.h>

namespace ktt
{

template <typename T>
const T* ValidationData::GetReferenceResult() const
{
    return reinterpret_cast<const T*>(m_ReferenceResult.data());
}

template <typename T>
const T* ValidationData::GetReferenceKernelResult() const
{
    return reinterpret_cast<const T*>(m_ReferenceKernelResult.data());
}

} // namespace ktt
