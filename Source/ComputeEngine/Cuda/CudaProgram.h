#pragma once

#ifdef KTT_API_CUDA

#include <string>
#include <nvrtc.h>

namespace ktt
{

class CudaProgram
{
public:
    explicit CudaProgram(const std::string& name, const std::string& source, const std::string& typeName = "");
    ~CudaProgram();

    void Build(const std::string& compilerOptions) const;

    const std::string& GetSource() const;
    std::string GetLoweredName() const;
    nvrtcProgram GetProgram() const;
    std::string GetPtxSource() const;

private:
    std::string m_Name;
    std::string m_Source;
    nvrtcProgram m_Program;

    std::string GetBuildInfo() const;
};

} // namespace ktt

#endif // KTT_API_CUDA
