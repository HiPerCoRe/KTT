#ifdef KTT_API_OPENCL

#include <ComputeEngine/OpenCl/OpenClKernel.h>
#include <ComputeEngine/OpenCl/OpenClUtility.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/ErrorHandling/KttException.h>

namespace ktt
{

OpenClKernel::OpenClKernel(std::unique_ptr<OpenClProgram> program, const std::string& name) :
    m_Name(name),
    m_Program(std::move(program)),
    m_NextArgumentIndex(0)
{
    KttAssert(m_Program != nullptr, "Invalid program was used during OpenCL kernel initialization");

    cl_int result;
    m_Kernel = clCreateKernel(m_Program->GetProgram(), m_Name.data(), &result);
    CheckError(result, "clCreateKernel");
}

OpenClKernel::~OpenClKernel()
{
    CheckError(clReleaseKernel(m_Kernel), "clReleaseKernel");
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

void OpenClKernel::ResetArguments()
{
    m_NextArgumentIndex = 0;
}

KernelCompilationData OpenClKernel::GenerateCompilationData() const
{
    KernelCompilationData result;

    result.m_MaxWorkGroupSize = GetAttribute(CL_KERNEL_WORK_GROUP_SIZE);
    result.m_LocalMemorySize = GetAttribute(CL_KERNEL_LOCAL_MEM_SIZE);
    result.m_PrivateMemorySize = GetAttribute(CL_KERNEL_PRIVATE_MEM_SIZE);
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

void OpenClKernel::SetKernelArgumentVector(const void* buffer)
{
    CheckError(clSetKernelArg(m_Kernel, m_NextArgumentIndex, sizeof(cl_mem), buffer), "clSetKernelArg");
    ++m_NextArgumentIndex;
}

void OpenClKernel::SetKernelArgumentVectorSVM([[maybe_unused]] const void* buffer)
{
#ifdef CL_VERSION_2_0
    CheckError(clSetKernelArgSVMPointer(m_Kernel, m_NextArgumentIndex, buffer), "clSetKernelArgSVMPointer");
    ++m_NextArgumentIndex;
#else
    throw KttException("Unified memory buffers are not supported on this platform");
#endif
}

void OpenClKernel::SetKernelArgumentScalar(const void* value, const size_t size)
{
    CheckError(clSetKernelArg(m_Kernel, m_NextArgumentIndex, size, value), "clSetKernelArg");
    ++m_NextArgumentIndex;
}

void OpenClKernel::SetKernelArgumentLocal(const size_t size)
{
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

} // namespace ktt

#endif // KTT_API_OPENCL
