#pragma once

namespace ktt
{

enum class CommandPriority
{
    TunerCreation,
    General,
    KernelCreation,
    ArgumentAddition,
    ParameterDefinition,
    ConstraintDefinition,
    ModifierDefinition,
    Tuning,
    Output
};

} // namespace ktt
