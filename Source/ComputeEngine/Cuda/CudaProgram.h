#pragma once

#ifdef KTT_API_CUDA

#include <string>
#include <nvrtc.h>

namespace ktt
{

class CudaContext;

class CudaProgram
{
public:
    explicit CudaProgram(const std::string& name, const std::string& source, const std::string& typeName = "");
    ~CudaProgram();

    void Build() const;

    const std::string& GetSource() const;
    std::string GetLoweredName() const;
    nvrtcProgram GetProgram() const;
    std::string GetPtxSource() const;

    static void InitializeCompilerOptions(const CudaContext& context);
    static void SetCompilerOptions(const std::string& options);

private:
    std::string m_Name;
    std::string m_Source;
    nvrtcProgram m_Program;

    inline static std::string m_CompilerOptions;

    std::string GetBuildInfo() const;
};

} // namespace ktt

#endif // KTT_API_CUDA
