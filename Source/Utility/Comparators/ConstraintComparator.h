#pragma once
#include <tuple>
#include <Kernel/KernelConstraint/KernelConstraint.h>

namespace ktt
{
struct ConstraintPointerComparator
{
    bool operator()(const KernelConstraint *a, const KernelConstraint *b) const
    {
        // Compare based on size and order together
        const size_t aSize = a->GetParameters().size();
        const size_t bSize = b->GetParameters().size();
        const auto aOrder = a->GetOrder();
        const auto bOrder = b->GetOrder();

        // Create a combined comparison using size as primary and order as secondary
        return std::tie(aSize, aOrder) < std::tie(bSize, bOrder);
    }
};

} // namespace ktt
