#ifdef KTT_API_OPENCL

#include <Api/Output/KernelCompilationData.h>
#include <Api/KttException.h>
#include <ComputeEngine/OpenCl/Actions/OpenClComputeAction.h>
#include <ComputeEngine/OpenCl/Buffers/OpenClBuffer.h>
#include <ComputeEngine/OpenCl/OpenClCommandQueue.h>
#include <ComputeEngine/OpenCl/OpenClKernel.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <KernelArgument/KernelArgument.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

OpenClKernel::OpenClKernel(std::unique_ptr<OpenClProgram> program, const std::string& name,
    IdGenerator<ComputeActionId>& generator) :
    m_Name(name),
    m_Program(std::move(program)),
    m_Generator(generator),
    m_NextArgumentIndex(0)
{
    Logger::LogDebug("Initializing OpenCL kernel with name " + name);
    KttAssert(m_Program != nullptr, "Invalid program was used during OpenCL kernel initialization");

    cl_int result;
    m_Kernel = clCreateKernel(m_Program->GetProgram(), m_Name.data(), &result);
    CheckError(result, "clCreateKernel");
}

OpenClKernel::~OpenClKernel()
{
    Logger::LogDebug("Releasing OpenCL kernel with name " + m_Name);
    CheckError(clReleaseKernel(m_Kernel), "clReleaseKernel");
}

std::unique_ptr<OpenClComputeAction> OpenClKernel::Launch(const OpenClCommandQueue& queue, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    const DimensionVector adjustedSize = AdjustGlobalSize(globalSize, localSize);
    const auto id = m_Generator.GenerateId();
    auto action = std::make_unique<OpenClComputeAction>(id, shared_from_this(), adjustedSize, localSize);

    Logger::LogDebug("Launching kernel " + m_Name + " with compute action id " + std::to_string(id) + ", global thread size: "
        + adjustedSize.GetString() + ", local thread size: " + localSize.GetString());
    const auto globalVector = adjustedSize.GetVector();
    const auto localVector = localSize.GetVector();

    cl_int result = clEnqueueNDRangeKernel(queue.GetQueue(), m_Kernel, static_cast<cl_uint>(globalVector.size()), nullptr,
        globalVector.data(), localVector.data(), 0, nullptr, action->GetEvent());
    CheckError(result, "clEnqueueNDRangeKernel");

    action->SetReleaseFlag();
    return action;
}

void OpenClKernel::SetArgument(const KernelArgument& argument)
{
    switch (argument.GetMemoryType())
    {
    case ArgumentMemoryType::Scalar:
        SetKernelArgumentScalar(argument.GetData(), argument.GetElementSize());
        break;
    case ArgumentMemoryType::Vector:
        KttError("Vector arguments have to be uploaded into buffer and set through corresponding method");
        break;
    case ArgumentMemoryType::Local:
        SetKernelArgumentLocal(argument.GetDataSize());
        break;
    default:
        KttError("Unhandled argument memory type value");
    }
}

void OpenClKernel::SetArgument(OpenClBuffer& buffer)
{
    switch (buffer.GetMemoryLocation())
    {
    case ArgumentMemoryLocation::Undefined:
        KttError("Non-vector arguments should be set through other argument setter method");
        break;
    case ArgumentMemoryLocation::Device:
    case ArgumentMemoryLocation::Host:
    case ArgumentMemoryLocation::HostZeroCopy:
        SetKernelArgumentVector(buffer.GetRawBuffer());
        break;
    case ArgumentMemoryLocation::Unified:
        SetKernelArgumentVectorSvm(buffer.GetRawBuffer());
        break;
    default:
        KttError("Unhandled argument memory location value");
    }
}

void OpenClKernel::ResetArguments()
{
    Logger::LogDebug("Resetting arguments for OpenCL kernel with name " + m_Name);
    m_NextArgumentIndex = 0;
}

std::unique_ptr<KernelCompilationData> OpenClKernel::GenerateCompilationData() const
{
    auto result = std::make_unique<KernelCompilationData>();

    result->m_MaxWorkGroupSize = GetAttribute(CL_KERNEL_WORK_GROUP_SIZE);
    result->m_LocalMemorySize = GetAttribute(CL_KERNEL_LOCAL_MEM_SIZE);
    result->m_PrivateMemorySize = GetAttribute(CL_KERNEL_PRIVATE_MEM_SIZE);
    // It is currently not possible to retrieve kernel constant memory size and registers count in OpenCL

    return result;
}

const std::string& OpenClKernel::GetName() const
{
    return m_Name;
}

cl_kernel OpenClKernel::GetKernel() const
{
    return m_Kernel;
}

void OpenClKernel::SetGlobalSizeType(const GlobalSizeType type)
{
    m_GlobalSizeType = type;
}

void OpenClKernel::SetGlobalSizeCorrection(const bool flag)
{
    m_GlobalSizeCorrection = flag;
}

void OpenClKernel::SetKernelArgumentVector(const void* buffer)
{
    Logger::LogDebug("Setting vector argument on index " + std::to_string(m_NextArgumentIndex)
        + " for OpenCL kernel with name " + m_Name);
    CheckError(clSetKernelArg(m_Kernel, m_NextArgumentIndex, sizeof(cl_mem), buffer), "clSetKernelArg");
    ++m_NextArgumentIndex;
}

void OpenClKernel::SetKernelArgumentVectorSvm([[maybe_unused]] const void* buffer)
{
#ifdef CL_VERSION_2_0
    Logger::LogDebug("Setting SVM vector argument on index " + std::to_string(m_NextArgumentIndex)
        + " for OpenCL kernel with name " + m_Name);
    CheckError(clSetKernelArgSVMPointer(m_Kernel, m_NextArgumentIndex, buffer), "clSetKernelArgSVMPointer");
    ++m_NextArgumentIndex;
#else
    throw KttException("Unified memory buffers are not supported on this platform");
#endif
}

void OpenClKernel::SetKernelArgumentScalar(const void* value, const size_t size)
{
    Logger::LogDebug("Setting scalar argument on index " + std::to_string(m_NextArgumentIndex)
        + " for OpenCL kernel with name " + m_Name);
    CheckError(clSetKernelArg(m_Kernel, m_NextArgumentIndex, size, value), "clSetKernelArg");
    ++m_NextArgumentIndex;
}

void OpenClKernel::SetKernelArgumentLocal(const size_t size)
{
    Logger::LogDebug("Setting local memory argument on index " + std::to_string(m_NextArgumentIndex)
        + " for OpenCL kernel with name " + m_Name);
    CheckError(clSetKernelArg(m_Kernel, m_NextArgumentIndex, size, nullptr), "clSetKernelArg");
    ++m_NextArgumentIndex;
}

uint64_t OpenClKernel::GetAttribute(const cl_kernel_work_group_info attribute) const
{
    uint64_t result;
    CheckError(clGetKernelWorkGroupInfo(m_Kernel, m_Program->GetDevice(), attribute, sizeof(result), &result, nullptr),
        "clGetKernelWorkGroupInfo");
    return result;
}

DimensionVector OpenClKernel::AdjustGlobalSize(const DimensionVector& globalSize, const DimensionVector& localSize)
{
    DimensionVector result = globalSize;

    switch (m_GlobalSizeType)
    {
    case GlobalSizeType::OpenCL:
        // Do nothing
        break;
    case GlobalSizeType::CUDA:
    case GlobalSizeType::Vulkan:
        result.Multiply(localSize);
        break;
    default:
        KttError("Unhandled global size type value");
        break;
    }

    if (m_GlobalSizeCorrection)
    {
        result.RoundUp(localSize);
    }

    return result;
}

} // namespace ktt

#endif // KTT_API_OPENCL
