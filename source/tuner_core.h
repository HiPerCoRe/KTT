#pragma once

#include <memory>

#include "kernel/kernel_manager.h"
#include "tuning_runner/tuning_runner.h"

namespace ktt
{

class TunerCore
{
public:
    // Constructor
    TunerCore();

    // Getters
    const std::shared_ptr<KernelManager> getKernelManager();
    const std::shared_ptr<TuningRunner> getTuningRunner();

private:
    // Attributes
    std::shared_ptr<KernelManager> kernelManager;
    std::shared_ptr<TuningRunner> tuningRunner;
};

} // namespace ktt
