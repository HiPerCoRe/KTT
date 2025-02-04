#pragma once

#ifdef KTT_API_CUDA

#include <memory>
#include <string>
#include <vector>
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
class EngineConfiguration;
class KernelArgument;
struct KernelCompilationData;

class CudaKernel : public std::enable_shared_from_this<CudaKernel>
{
public:
    explicit CudaKernel(IdGenerator<ComputeActionId>& generator, const EngineConfiguration& configuration, const std::string& name,
        const std::string& source, const std::string& templatedName = "", const std::vector<KernelArgument*>& symbolArguments = {});
    ~CudaKernel();

    std::unique_ptr<CudaComputeAction> Launch(const CudaStream& stream, const DimensionVector& globalSize,
        const DimensionVector& localSize, const std::vector<CUdeviceptr*>& arguments, const size_t sharedMemorySize);
    std::unique_ptr<KernelCompilationData> GenerateCompilationData() const;

    const std::string& GetName() const;
    CUfunction GetKernel() const;
    CUmodule GetModule() const;
    uint64_t GetAttribute(const CUfunction_attribute attribute) const;

private:
    std::string m_Name;
    std::unique_ptr<CudaProgram> m_Program;
    IdGenerator<ComputeActionId>& m_Generator;
    const EngineConfiguration& m_Configuration;
    CUfunction m_Kernel;
    CUmodule m_Module;

    DimensionVector AdjustGlobalSize(const DimensionVector& globalSize, const DimensionVector& localSize) const;
};

} // namespace ktt

#endif // KTT_API_CUDA
