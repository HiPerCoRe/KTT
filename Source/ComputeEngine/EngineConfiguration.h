#pragma once

#include <string>

#include <ComputeEngine/GlobalSizeType.h>

namespace ktt
{

class EngineConfiguration
{
public:
    EngineConfiguration();
    explicit EngineConfiguration(const GlobalSizeType sizeType);

    void SetStaticCompilerOptions(const std::string& options);
    void SetTuningCompilerOptions(const std::string& options);
    void SetGlobalSizeType(const GlobalSizeType sizeType);
    void SetGlobalSizeCorrection(const bool sizeCorrection);

    std::string GetCompilerOptions() const;
    std::string GetStaticCompilerOptions() const;
    GlobalSizeType GetGlobalSizeType() const;
    bool GetGlobalSizeCorrection() const;

    bool IsProfilingActive() const;
    void SetProfiling(const bool profiling);

private:
    std::string m_StaticCompilerOptions;
    std::string m_TuningCompilerOptions;
    GlobalSizeType m_GlobalSizeType;
    bool m_GlobalSizeCorrection;
    bool m_ProfilingFlag;
};

} // namespace ktt
