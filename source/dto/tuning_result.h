#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../kernel/kernel_configuration.h"

namespace ktt
{

class TuningResult
{
public:
    explicit TuningResult(const std::string& kernelName, const uint64_t kernelDuration, const KernelConfiguration& configuration);
    explicit TuningResult(const std::string& kernelName, const uint64_t kernelDuration, const uint64_t manipulatorDuration,
        const KernelConfiguration& configuration);

    std::string getKernelName() const;
    uint64_t getKernelDuration() const;
    uint64_t getManipulatorDuration() const;
    uint64_t getTotalDuration() const;
    KernelConfiguration getConfiguration() const;

    friend std::ostream& operator<<(std::ostream& outputTarget, const TuningResult& tuningResult);

private:
    std::string kernelName;
    uint64_t kernelDuration;
    uint64_t manipulatorDuration;
    KernelConfiguration configuration;
};

std::ostream& operator<<(std::ostream& outputTarget, const TuningResult& tuningResult);

} // namespace ktt
