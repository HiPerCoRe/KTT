#pragma once

namespace ktt
{

enum class CommandPriority
{
    TunerCreation,
    General,
    KernelCreation,
    ParameterDefinition,
    ConstraintDefinition,
    Tuning,
    Output
};

} // namespace ktt
