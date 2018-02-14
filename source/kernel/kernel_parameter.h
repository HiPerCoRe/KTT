#pragma once

#include <string>
#include <utility>
#include <vector>
#include "ktt_types.h"
#include "enum/modifier_action.h"
#include "enum/modifier_dimension.h"
#include "enum/modifier_type.h"

namespace ktt
{

class KernelParameter
{
public:
    explicit KernelParameter(const std::string& name, const std::vector<size_t>& values, const ModifierType modifierType,
        const ModifierAction modifierAction, const ModifierDimension modifierDimension);
    explicit KernelParameter(const std::string& name, const std::vector<double>& values);

    void setLocalMemoryArgumentModifier(const ArgumentId id, const ModifierAction modifierAction);
    void setLocalMemoryArgumentModifier(const KernelId compositionKernelId, ArgumentId id, const ModifierAction modifierAction);
    void addCompositionKernel(const KernelId id);

    std::string getName() const;
    std::vector<size_t> getValues() const;
    std::vector<double> getValuesDouble() const;
    ModifierType getModifierType() const;
    ModifierAction getModifierAction() const;
    ModifierDimension getModifierDimension() const;
    std::vector<KernelId> getCompositionKernels() const;
    bool hasValuesDouble() const;
    bool isLocalMemoryModifier() const;
    std::vector<std::pair<ArgumentId, ModifierAction>> getLocalMemoryArguments() const;
    std::vector<KernelId> getLocalMemoryModifierKernels() const;

    bool operator==(const KernelParameter& other) const;
    bool operator!=(const KernelParameter& other) const;

private:
    std::string name;
    std::vector<size_t> values;
    std::vector<double> valuesDouble;
    ModifierType modifierType;
    ModifierAction modifierAction;
    ModifierDimension modifierDimension;
    std::vector<KernelId> compositionKernels;
    bool isDouble;
    bool localMemoryModifierFlag;
    std::vector<std::pair<ArgumentId, ModifierAction>> localMemoryArguments;
    std::vector<KernelId> localMemoryModifierKernels;
};

} // namespace ktt
