#pragma once

#include <memory>

#include "kernel/kernel_manager.h"

namespace ktt
{

class TunerCore
{
public:
    // Constructor
    TunerCore();

    // Core methods
    // none

    // Getters
    const std::shared_ptr<KernelManager> getKernelManager();

private:
    // Attributes
    std::shared_ptr<KernelManager> kernelManager;

    // Helper methods
    // none
};

} // namespace ktt
