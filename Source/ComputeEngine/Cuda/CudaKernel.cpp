#ifdef KTT_API_CUDA

#include <Api/Output/KernelCompilationData.h>
#include <ComputeEngine/Cuda/CudaKernel.h>
#include <ComputeEngine/cuda/CudaUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaKernel::CudaKernel(std::unique_ptr<CudaProgram> program, const std::string& name, ActionIdGenerator& generator) :
    m_Name(name),
    m_Program(std::move(program)),
    m_Generator(generator)
{
    Logger::LogDebug("Initializing CUDA kernel with name " + name);
    KttAssert(m_Program != nullptr, "Invalid program was used during CUDA kernel initialization");

    const std::string ptx = m_Program->GetPtxSource();
    CheckError(cuModuleLoadDataEx(&m_Module, ptx.data(), 0, nullptr, nullptr), "cuModuleLoadDataEx");
    CheckError(cuModuleGetFunction(&m_Kernel, m_Module, name.data()), "cuModuleGetFunction");
}

CudaKernel::~CudaKernel()
{
    Logger::LogDebug("Releasing CUDA kernel with name " + m_Name);
    CheckError(cuModuleUnload(m_Module), "cuModuleUnload");
}

std::unique_ptr<KernelCompilationData> CudaKernel::GenerateCompilationData() const
{
    auto result = std::make_unique<KernelCompilationData>();

    result->m_MaxWorkGroupSize = GetAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    result->m_LocalMemorySize = GetAttribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
    result->m_PrivateMemorySize = GetAttribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
    result->m_ConstantMemorySize = GetAttribute(CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
    result->m_RegistersCount = GetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS);

    return result;
}

const std::string& CudaKernel::GetName() const
{
    return m_Name;
}

CUfunction CudaKernel::GetKernel() const
{
    return m_Kernel;
}

CUmodule CudaKernel::GetModule() const
{
    return m_Module;
}

void CudaKernel::SetGlobalSizeType(const GlobalSizeType type)
{
    m_GlobalSizeType = type;
}

void CudaKernel::SetGlobalSizeCorrection(const bool flag)
{
    m_GlobalSizeCorrection = flag;
}

uint64_t CudaKernel::GetAttribute(const CUfunction_attribute attribute) const
{
    int value;
    CheckError(cuFuncGetAttribute(&value, attribute, m_Kernel), "cuFuncGetAttribute");
    return static_cast<uint64_t>(value);
}

} // namespace ktt

#endif // KTT_API_CUDA
