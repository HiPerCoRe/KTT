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
    explicit TuningResult(const uint64_t duration, const KernelConfiguration& configuration);

    uint64_t getDuration() const;
    KernelConfiguration getConfiguration() const;

    friend std::ostream& operator<<(std::ostream& outputTarget, const TuningResult& tuningResult);

private:
    uint64_t duration;
    KernelConfiguration configuration;
};

std::ostream& operator<<(std::ostream& outputTarget, const TuningResult& tuningResult);

} // namespace ktt
