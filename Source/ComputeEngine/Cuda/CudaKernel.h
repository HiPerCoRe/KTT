#pragma once

#ifdef KTT_API_CUDA

#include <memory>
#include <string>
#include <cuda.h>

#include <Api/Configuration/DimensionVector.h>
#include <ComputeEngine/Cuda/CudaProgram.h>
#include <ComputeEngine/GlobalSizeType.h>
#include <Utility/IdGenerator.h>
#include <KttTypes.h>

namespace ktt
{

class CudaComputeAction;
class CudaStream;
struct KernelCompilationData;

class CudaKernel : public std::enable_shared_from_this<CudaKernel>
{
public:
    explicit CudaKernel(IdGenerator<ComputeActionId>& generator, const std::string& name, const std::string& source,
        const std::string& typeName = "");
    ~CudaKernel();

    std::unique_ptr<CudaComputeAction> Launch(const CudaStream& stream, const DimensionVector& globalSize,
        const DimensionVector& localSize, const std::vector<CUdeviceptr*>& arguments, const size_t sharedMemorySize);
    std::unique_ptr<KernelCompilationData> GenerateCompilationData() const;

    const std::string& GetName() const;
    CUfunction GetKernel() const;
    CUmodule GetModule() const;

    static void SetGlobalSizeType(const GlobalSizeType type);
    static void SetGlobalSizeCorrection(const bool flag);

private:
    std::string m_Name;
    std::unique_ptr<CudaProgram> m_Program;
    IdGenerator<ComputeActionId>& m_Generator;
    CUfunction m_Kernel;
    CUmodule m_Module;

    inline static GlobalSizeType m_GlobalSizeType = GlobalSizeType::CUDA;
    inline static bool m_GlobalSizeCorrection = false;

    uint64_t GetAttribute(const CUfunction_attribute attribute) const;
    static DimensionVector AdjustGlobalSize(const DimensionVector& globalSize, const DimensionVector& localSize);
};

} // namespace ktt

#endif // KTT_API_CUDA
