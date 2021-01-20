#pragma once

#ifdef KTT_API_CUDA

#include <memory>
#include <string>
#include <cuda.h>

#include <ComputeEngine/Cuda/CudaProgram.h>
#include <ComputeEngine/ActionIdGenerator.h>
#include <ComputeEngine/GlobalSizeType.h>

namespace ktt
{

struct KernelCompilationData;

class CudaKernel : public std::enable_shared_from_this<CudaKernel>
{
public:
    explicit CudaKernel(std::unique_ptr<CudaProgram> program, const std::string& name, ActionIdGenerator& generator);
    ~CudaKernel();

    std::unique_ptr<KernelCompilationData> GenerateCompilationData() const;

    const std::string& GetName() const;
    CUfunction GetKernel() const;
    CUmodule GetModule() const;

    static void SetGlobalSizeType(const GlobalSizeType type);
    static void SetGlobalSizeCorrection(const bool flag);

private:
    std::string m_Name;
    std::unique_ptr<CudaProgram> m_Program;
    ActionIdGenerator& m_Generator;
    CUfunction m_Kernel;
    CUmodule m_Module;

    inline static GlobalSizeType m_GlobalSizeType = GlobalSizeType::CUDA;
    inline static bool m_GlobalSizeCorrection = false;

    uint64_t GetAttribute(const CUfunction_attribute attribute) const;
};

} // namespace ktt

#endif // KTT_API_CUDA
