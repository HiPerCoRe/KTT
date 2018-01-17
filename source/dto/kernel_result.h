#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "kernel/kernel_configuration.h"

namespace ktt
{

class KernelResult
{
public:
    KernelResult();
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration);
    explicit KernelResult(const std::string& kernelName, uint64_t kernelDuration);
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& errorMessage);

    void setKernelName(const std::string& kernelName);
    void setConfiguration(const KernelConfiguration& configuration);
    void setKernelDuration(const uint64_t kernelDuration);
    void setManipulatorDuration(const uint64_t manipulatorDuration);
    void setOverhead(const uint64_t overhead);
    void setErrorMessage(const std::string& errorMessage);
    void setValid(const bool flag);

    std::string getKernelName() const;
    KernelConfiguration getConfiguration() const;
    uint64_t getKernelDuration() const;
    uint64_t getManipulatorDuration() const;
    uint64_t getOverhead() const;
    uint64_t getTotalDuration() const;
    std::string getErrorMessage() const;
    bool isValid() const;

private:
    std::string kernelName;
    KernelConfiguration configuration;
    uint64_t kernelDuration;
    uint64_t manipulatorDuration;
    uint64_t overhead;
    std::string errorMessage;
    bool valid;
};

} // namespace ktt
