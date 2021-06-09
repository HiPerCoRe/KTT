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

    void SetCompilerOptions(const std::string& options);
    void SetGlobalSizeType(const GlobalSizeType sizeType);
    void SetGlobalSizeCorrection(const bool sizeCorrection);

    const std::string& GetCompilerOptions() const;
    GlobalSizeType GetGlobalSizeType() const;
    bool GetGlobalSizeCorrection() const;

private:
    std::string m_CompilerOptions;
    GlobalSizeType m_GlobalSizeType;
    bool m_GlobalSizeCorrection;
};

} // namespace ktt
