#pragma once

#ifdef KTT_API_CUDA

#include <string>
#include <vector>
#include <nvrtc.h>

namespace ktt
{

class CudaKernel;
class KernelArgument;

class CudaProgram
{
public:
    explicit CudaProgram(const std::string& name, const std::string& source, const std::vector<KernelArgument*>& symbolArguments = {});
    ~CudaProgram();

    void Build(const std::string& compilerOptions) const;
    void InitializeSymbolData(const CudaKernel& kernel) const;

    const std::string& GetSource() const;
    std::string GetLoweredName() const;
    nvrtcProgram GetProgram() const;
    std::string GetPtxSource() const;

private:
    std::string m_Name;
    std::string m_Source;
    std::vector<KernelArgument*> m_SymbolArguments;
    nvrtcProgram m_Program;

    std::string GetBuildInfo() const;
};

} // namespace ktt

#endif // KTT_API_CUDA
