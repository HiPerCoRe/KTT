#include "tuner_core.h"

namespace ktt
{

TunerCore::TunerCore():
    kernelManager(new KernelManager())
{}

const std::shared_ptr<KernelManager> TunerCore::getKernelManager()
{
    return kernelManager;
}

} // namespace ktt
