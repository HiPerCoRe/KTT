#pragma once

#include <vector>

#include "../enums/dimension.h"
#include "../enums/thread_modifier_action.h"
#include "../enums/thread_modifier_type.h"

namespace ktt
{

class KernelParameter
{
public:
    explicit KernelParameter(const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
        const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension):
        name(name),
        values(values),
        threadModifierType(threadModifierType),
        threadModifierAction(threadModifierAction),
        modifierDimension(modifierDimension)
    {}
    
    std::string getName() const
    {
        return name;
    }

    std::vector<size_t> getValues() const
    {
        return values;
    }

    ThreadModifierType getThreadModifierType() const
    {
        return threadModifierType;
    }

    ThreadModifierAction getThreadModifierAction() const
    {
        return threadModifierAction;
    }

    Dimension getModifierDimension() const
    {
        return modifierDimension;
    }

private:
    std::string name;
    std::vector<size_t> values;
    ThreadModifierType threadModifierType;
    ThreadModifierAction threadModifierAction;
    Dimension modifierDimension;
};

} // namespace ktt
