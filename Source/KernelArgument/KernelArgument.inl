#include <cstdint>
#include <vector>

#include <KernelArgument/KernelArgument.h>

namespace ktt
{

template <typename T>
const T* KernelArgument::GetDataWithType() const
{
    return reinterpret_cast<const T*>(GetData());
}

template <typename T>
uint64_t KernelArgument::GetNumberOfElementsWithType() const
{
    return GetDataSize() / sizeof(T);
}

} // namespace ktt
