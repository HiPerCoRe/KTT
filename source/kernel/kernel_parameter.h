#pragma once

#include <string>
#include <vector>

#include "enum/dimension.h"
#include "enum/thread_modifier_action.h"
#include "enum/thread_modifier_type.h"

namespace ktt
{

class KernelParameter
{
public:
    explicit KernelParameter(const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& threadModifierType,
        const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension);
    
    std::string getName() const;
    std::vector<size_t> getValues() const;
    ThreadModifierType getThreadModifierType() const;
    ThreadModifierAction getThreadModifierAction() const;
    Dimension getModifierDimension() const;

    bool operator==(const KernelParameter& other) const;
    bool operator!=(const KernelParameter& other) const;

private:
    std::string name;
    std::vector<size_t> values;
    ThreadModifierType threadModifierType;
    ThreadModifierAction threadModifierAction;
    Dimension modifierDimension;
};

} // namespace ktt
