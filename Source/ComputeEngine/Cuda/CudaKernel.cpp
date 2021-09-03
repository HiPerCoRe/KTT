#ifdef KTT_API_CUDA

#include <Api/Output/KernelCompilationData.h>
#include <ComputeEngine/Cuda/Actions/CudaComputeAction.h>
#include <ComputeEngine/Cuda/CudaKernel.h>
#include <ComputeEngine/Cuda/CudaStream.h>
#include <ComputeEngine/Cuda/CudaUtility.h>
#include <ComputeEngine/EngineConfiguration.h>
#include <KernelArgument/KernelArgument.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

CudaKernel::CudaKernel(IdGenerator<ComputeActionId>& generator, const EngineConfiguration& configuration, const std::string& name,
    const std::string& source, const std::string& templatedName, const std::vector<KernelArgument*>& symbolArguments) :
    m_Name(name),
    m_Generator(generator),
    m_Configuration(configuration)
{
    Logger::LogDebug("Initializing CUDA kernel with name " + name);

    const auto& programName = templatedName.empty() ? name : templatedName;
    m_Program = std::make_unique<CudaProgram>(programName, source, symbolArguments);
    m_Program->Build(m_Configuration.GetCompilerOptions());

    const std::string ptx = m_Program->GetPtxSource();
    CheckError(cuModuleLoadDataEx(&m_Module, ptx.data(), 0, nullptr, nullptr), "cuModuleLoadDataEx");

    const std::string loweredName = m_Program->GetLoweredName();
    CheckError(cuModuleGetFunction(&m_Kernel, m_Module, loweredName.c_str()), "cuModuleGetFunction");

    m_Program->InitializeSymbolData(*this);
}

CudaKernel::~CudaKernel()
{
    Logger::LogDebug("Releasing CUDA kernel with name " + m_Name);
    CheckError(cuModuleUnload(m_Module), "cuModuleUnload");
}

std::unique_ptr<CudaComputeAction> CudaKernel::Launch(const CudaStream& stream, const DimensionVector& globalSize,
    const DimensionVector& localSize, const std::vector<CUdeviceptr*>& arguments, const size_t sharedMemorySize)
{
    std::vector<void*> kernelArguments;

    for (size_t i = 0; i < arguments.size(); ++i)
    {
        kernelArguments.push_back(static_cast<void*>(arguments[i]));
    }

    const DimensionVector adjustedSize = AdjustGlobalSize(globalSize, localSize);
    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<CudaComputeAction>(id, shared_from_this(), adjustedSize, localSize);

    Logger::LogDebug("Launching kernel " + m_Name + " with compute action id " + std::to_string(id) + ", global thread size: "
        + adjustedSize.GetString() + ", local thread size: " + localSize.GetString());
    const auto globalVector = adjustedSize.GetVector();
    const auto localVector = localSize.GetVector();

    CheckError(cuEventRecord(action->GetStartEvent(), stream.GetStream()), "cuEventRecord");
    CheckError(cuLaunchKernel(m_Kernel, static_cast<unsigned int>(globalVector[0]), static_cast<unsigned int>(globalVector[1]),
        static_cast<unsigned int>(globalVector[2]), static_cast<unsigned int>(localVector[0]),
        static_cast<unsigned int>(localVector[1]), static_cast<unsigned int>(localVector[2]),
        static_cast<unsigned int>(sharedMemorySize), stream.GetStream(), kernelArguments.data(), nullptr), "cuLaunchKernel");
    CheckError(cuEventRecord(action->GetEndEvent(), stream.GetStream()), "cuEventRecord");

    return action;
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

uint64_t CudaKernel::GetAttribute(const CUfunction_attribute attribute) const
{
    int value;
    CheckError(cuFuncGetAttribute(&value, attribute, m_Kernel), "cuFuncGetAttribute");
    return static_cast<uint64_t>(value);
}

DimensionVector CudaKernel::AdjustGlobalSize(const DimensionVector& globalSize, const DimensionVector& localSize) const
{
    DimensionVector result = globalSize;

    if (m_Configuration.GetGlobalSizeCorrection())
    {
        if (m_Configuration.GetGlobalSizeType() == GlobalSizeType::OpenCL)
        {
            result.RoundUp(localSize);
        }
        else
        {
            Logger::LogWarning("Global size correction has no effect if global size type is not set to OpenCL");
        }
    }

    switch (m_Configuration.GetGlobalSizeType())
    {
    case GlobalSizeType::OpenCL:
        result.Divide(localSize);
        break;
    case GlobalSizeType::CUDA:
    case GlobalSizeType::Vulkan:
        // Do nothing
        break;
    default:
        KttError("Unhandled global size type value");
        break;
    }

    return result;
}

} // namespace ktt

#endif // KTT_API_CUDA
