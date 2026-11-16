#pragma once
#include <KernelParameter.h>


namespace ktt
{
struct CompareParametersByName
{
    bool operator()(const KernelParameter *a, const KernelParameter *b) const
    {
        return a->GetName() < b->GetName();
    }
};

} // namespace ktt
