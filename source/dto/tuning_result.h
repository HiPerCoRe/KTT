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
    explicit TuningResult(const std::string& kernelName, const uint64_t duration, const KernelConfiguration& configuration);

    std::string getKernelName() const;
    uint64_t getDuration() const;
    KernelConfiguration getConfiguration() const;

    friend std::ostream& operator<<(std::ostream& outputTarget, const TuningResult& tuningResult);

private:
    std::string kernelName;
    uint64_t duration;
    KernelConfiguration configuration;
};

std::ostream& operator<<(std::ostream& outputTarget, const TuningResult& tuningResult);

} // namespace ktt
