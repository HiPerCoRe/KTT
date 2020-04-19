#pragma once

#include <cstddef>
#include <functional>
#include <vector>
#include <ktt_types.h>

namespace ktt
{

class LocalMemoryModifier
{
public:
    LocalMemoryModifier();
    explicit LocalMemoryModifier(const KernelId kernel, const ArgumentId argument, const std::vector<size_t>& parameterValues,
        const std::function<size_t(const size_t, const std::vector<size_t>&)>& modifierFunction);

    KernelId getKernel() const;
    ArgumentId getArgument() const;
    const std::vector<size_t>& getParameterValues() const;
    std::function<size_t(const size_t, const std::vector<size_t>&)> getModifierFunction() const;
    size_t getModifiedSize(const size_t defaultSize) const;

private:
    KernelId kernel;
    ArgumentId argument;
    std::vector<size_t> parameterValues;
    std::function<size_t(const size_t, const std::vector<size_t>&)> modifierFunction;
};

} // namespace ktt
