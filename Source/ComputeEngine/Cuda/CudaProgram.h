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
    explicit CudaProgram(const std::string& source);
    ~CudaProgram();

    void Build() const;

    const std::string& GetSource() const;
    nvrtcProgram GetProgram() const;
    std::string GetPtxSource() const;

    static void InitializeCompilerOptions(const CudaContext& context);
    static void SetCompilerOptions(const std::string& options);

private:
    std::string m_Source;
    nvrtcProgram m_Program;

    inline static std::string m_CompilerOptions;

    std::string GetBuildInfo() const;
};

} // namespace ktt

#endif // KTT_API_CUDA
