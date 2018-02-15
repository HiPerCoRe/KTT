#pragma once

#include <cstddef>
#include "ktt_types.h"
#include "enum/modifier_action.h"

namespace ktt
{

class LocalMemoryModifier
{
public:
    LocalMemoryModifier();
    explicit LocalMemoryModifier(const KernelId kernel, const ArgumentId argument, const ModifierAction action, const size_t value);

    void setAction(const ModifierAction action);
    void setValue(const size_t value);

    KernelId getKernel() const;
    ArgumentId getArgument() const;
    ModifierAction getAction() const;
    size_t getValue() const;
    size_t getModifiedValue(const size_t value) const;

private:
    KernelId kernel;
    ArgumentId argument;
    ModifierAction action;
    size_t value;
};

} // namespace ktt
