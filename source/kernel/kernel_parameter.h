#pragma once

#include <string>
#include <vector>
#include "ktt_types.h"
#include "enum/dimension.h"
#include "enum/thread_modifier_action.h"
#include "enum/thread_modifier_type.h"

namespace ktt
{

class KernelParameter
{
public:
    explicit KernelParameter(const std::string& name, const std::vector<size_t>& values, const ThreadModifierType& modifierType,
        const ThreadModifierAction& modifierAction, const Dimension& modifierDimension);

    void addCompositionKernel(const KernelId id);

    std::string getName() const;
    std::vector<size_t> getValues() const;
    ThreadModifierType getModifierType() const;
    ThreadModifierAction getModifierAction() const;
    Dimension getModifierDimension() const;
    std::vector<KernelId> getCompositionKernels() const;

    bool operator==(const KernelParameter& other) const;
    bool operator!=(const KernelParameter& other) const;

private:
    std::string name;
    std::vector<size_t> values;
    ThreadModifierType threadModifierType;
    ThreadModifierAction threadModifierAction;
    Dimension modifierDimension;
    std::vector<KernelId> compositionKernels;
};

} // namespace ktt
