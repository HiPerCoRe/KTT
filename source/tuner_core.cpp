#include "tuner_core.h"

namespace ktt
{

TunerCore::TunerCore():
    kernelManager(new KernelManager()),
    tuningRunner(new TuningRunner())
{}

const std::shared_ptr<KernelManager> TunerCore::getKernelManager()
{
    return kernelManager;
}

const std::shared_ptr<TuningRunner> TunerCore::getTuningRunner()
{
    return tuningRunner;
}

} // namespace ktt
