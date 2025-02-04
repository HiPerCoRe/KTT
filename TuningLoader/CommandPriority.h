#pragma once

namespace ktt
{

enum class CommandPriority
{
    Logging,
    Initialization,
    General,
    Kernel,
    KernelArgument,
    SharedMemory,
    TuningParameter,
    Constraint,
    Modifier,
    Searcher,
    StopCondition,
    Validation,
    Tuning,
    Output
};

} // namespace ktt
