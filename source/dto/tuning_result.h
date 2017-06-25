#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "kernel/kernel_configuration.h"

namespace ktt
{

class TuningResult
{
public:
    explicit TuningResult(const std::string& kernelName, const uint64_t kernelDuration, const KernelConfiguration& configuration);
    explicit TuningResult(const std::string& kernelName, const uint64_t kernelDuration, const uint64_t manipulatorDuration,
        const KernelConfiguration& configuration);
    explicit TuningResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& statusMessage);

    std::string getKernelName() const;
    uint64_t getKernelDuration() const;
    uint64_t getManipulatorDuration() const;
    uint64_t getTotalDuration() const;
    KernelConfiguration getConfiguration() const;
    bool isValid() const;
    std::string getStatusMessage() const;

private:
    std::string kernelName;
    uint64_t kernelDuration;
    uint64_t manipulatorDuration;
    KernelConfiguration configuration;
    bool valid;
    std::string statusMessage;
};

} // namespace ktt
