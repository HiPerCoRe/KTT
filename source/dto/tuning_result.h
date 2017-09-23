#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "kernel_run_result.h"
#include "kernel/kernel_configuration.h"

namespace ktt
{

class TuningResult
{
public:
    explicit TuningResult(const std::string& kernelName, const KernelConfiguration& configuration);
    explicit TuningResult(const std::string& kernelName, const KernelConfiguration& configuration, const KernelRunResult& kernelRunResult);
    explicit TuningResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& statusMessage);

    void setKernelDuration(const uint64_t kernelDuration);
    void setKernelOverhead(const uint64_t kernelOverhead);
    void setManipulatorDuration(const uint64_t manipulatorDuration);
    void setValid(const bool flag);
    void setStatusMessage(const std::string& statusMessage);

    std::string getKernelName() const;
    KernelConfiguration getConfiguration() const;
    uint64_t getKernelDuration() const;
    uint64_t getKernelOverhead() const;
    uint64_t getManipulatorDuration() const;
    uint64_t getTotalDuration() const;
    bool isValid() const;
    std::string getStatusMessage() const;

private:
    std::string kernelName;
    KernelConfiguration configuration;
    uint64_t kernelDuration;
    uint64_t kernelOverhead;
    uint64_t manipulatorDuration;
    bool valid;
    std::string statusMessage;
};

} // namespace ktt
