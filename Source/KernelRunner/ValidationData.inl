#include <KernelRunner/ValidationData.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

template <typename T>
const T* ValidationData::GetReferenceResult() const
{
    if (HasReferenceComputation())
    {
        return GetReferenceComputationResult<T>();
    }

    if (HasReferenceKernel())
    {
        return GetReferenceKernelResult<T>();
    }
    
    if (HasReferenceArgument())
    {
        return GetReferenceArgumentResult<T>();
    }

    KttError("Unhandled validation case");
    return nullptr;
}

template <typename T>
const T* ValidationData::GetReferenceComputationResult() const
{
    return reinterpret_cast<const T*>(m_ReferenceResult.data());
}

template <typename T>
const T* ValidationData::GetReferenceKernelResult() const
{
    return reinterpret_cast<const T*>(m_ReferenceKernelResult.data());
}

template <typename T>
const T* ValidationData::GetReferenceArgumentResult() const
{
    return m_ReferenceArgument->GetDataWithType<T>();
}

} // namespace ktt
