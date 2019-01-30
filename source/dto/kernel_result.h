#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <api/kernel_profiling_data.h>
#include <kernel/kernel_configuration.h>

namespace ktt
{

class KernelResult
{
public:
    KernelResult();
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration);
    explicit KernelResult(const std::string& kernelName, uint64_t computationDuration);
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& errorMessage);

    void setKernelName(const std::string& kernelName);
    void setConfiguration(const KernelConfiguration& configuration);
    void setComputationDuration(const uint64_t computationDuration);
    void setOverhead(const uint64_t overhead);
    void setErrorMessage(const std::string& errorMessage);
    void setProfilingData(const KernelProfilingData& profilingData);
    void setCompositionKernelProfilingData(const KernelId id, const KernelProfilingData& profilingData);
    void setValid(const bool flag);

    const std::string& getKernelName() const;
    const KernelConfiguration& getConfiguration() const;
    uint64_t getComputationDuration() const;
    uint64_t getOverhead() const;
    const std::string& getErrorMessage() const;
    const KernelProfilingData& getProfilingData() const;
    const KernelProfilingData& getCompositionKernelProfilingData(const KernelId id) const;
    const std::map<KernelId, KernelProfilingData>& getCompositionProfilingData() const;
    bool isValid() const;

    void increaseOverhead(const uint64_t overhead);

private:
    std::string kernelName;
    KernelConfiguration configuration;
    uint64_t computationDuration;
    uint64_t overhead;
    std::string errorMessage;
    KernelProfilingData profilingData;
    std::map<KernelId, KernelProfilingData> compositionProfilingData;
    bool valid;
};

} // namespace ktt
