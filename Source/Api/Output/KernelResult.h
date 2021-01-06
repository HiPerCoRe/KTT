//#pragma once
//
//#include <cstdint>
//#include <map>
//#include <string>
//#include <vector>
//#include <api/computation_result.h>
//#include <api/kernel_compilation_data.h>
//#include <api/kernel_configuration.h>
//#include <api/kernel_profiling_data.h>
//
//namespace ktt
//{
//
//class KernelResult
//{
//public:
//    KernelResult();
//    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration);
//    explicit KernelResult(const std::string& kernelName, uint64_t computationDuration);
//    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& errorMessage);
//
//    void setKernelName(const std::string& kernelName);
//    void setConfiguration(const KernelConfiguration& configuration);
//    void setComputationDuration(const uint64_t computationDuration);
//    void setOverhead(const uint64_t overhead);
//    void setKernelTime(const uint64_t kernelTime);
//    void setErrorMessage(const std::string& errorMessage);
//    void setCompilationData(const KernelCompilationData& compilationData);
//    void setCompositionKernelCompilationData(const KernelId id, const KernelCompilationData& compilationData);
//    void setProfilingData(const KernelProfilingData& profilingData);
//    void setCompositionKernelProfilingData(const KernelId id, const KernelProfilingData& profilingData);
//    void setValid(const bool flag);
//
//    const std::string& getKernelName() const;
//    const KernelConfiguration& getConfiguration() const;
//    uint64_t getComputationDuration() const;
//    uint64_t getOverhead() const;
//    uint64_t getKernelTime() const;
//    const std::string& getErrorMessage() const;
//    const KernelCompilationData& getCompilationData() const;
//    const KernelCompilationData& getCompositionKernelCompilationData(const KernelId id) const;
//    const std::map<KernelId, KernelCompilationData>& getCompositionCompilationData() const;
//    const KernelProfilingData& getProfilingData() const;
//    const KernelProfilingData& getCompositionKernelProfilingData(const KernelId id) const;
//    const std::map<KernelId, KernelProfilingData>& getCompositionProfilingData() const;
//    bool isValid() const;
//
//    void increaseOverhead(const uint64_t overhead);
//    void increaseKernelTime(const uint64_t kernelTime);
//
//    ComputationResult getComputationResult() const;
//
//private:
//    std::string kernelName;
//    KernelConfiguration configuration;
//    uint64_t computationDuration;
//    uint64_t overhead;
//    uint64_t kernelTime;
//    std::string errorMessage;
//    KernelCompilationData compilationData;
//    std::map<KernelId, KernelCompilationData> compositionCompilationData;
//    KernelProfilingData profilingData;
//    std::map<KernelId, KernelProfilingData> compositionProfilingData;
//    bool valid;
//};
//
//} // namespace ktt
