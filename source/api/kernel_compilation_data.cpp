#include <api/kernel_compilation_data.h>

namespace ktt
{

KernelCompilationData::KernelCompilationData() :
    maxWorkGroupSize(0),
    localMemorySize(0),
    privateMemorySize(0),
    constantMemorySize(0),
    registersCount(0)
{}

} // namespace ktt
