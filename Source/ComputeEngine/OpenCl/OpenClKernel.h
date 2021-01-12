#pragma once

#ifdef KTT_API_OPENCL

#include <memory>
#include <string>
#include <CL/cl.h>

#include <Api/Output/KernelCompilationData.h>
#include <ComputeEngine/OpenCl/OpenClProgram.h>
#include <KernelArgument/KernelArgument.h>

namespace ktt
{

class OpenClKernel
{
public:
    explicit OpenClKernel(std::unique_ptr<OpenClProgram> program, const std::string& name);
    ~OpenClKernel();

    void SetArgument(const KernelArgument& argument);
    void ResetArguments();
    KernelCompilationData GenerateCompilationData() const;

    const std::string& GetName() const;
    cl_kernel GetKernel() const;

private:
    std::string m_Name;
    std::unique_ptr<OpenClProgram> m_Program;
    cl_kernel m_Kernel;
    cl_uint m_NextArgumentIndex;

    void SetKernelArgumentVector(const void* buffer);
    void SetKernelArgumentVectorSVM(const void* buffer);
    void SetKernelArgumentScalar(const void* value, const size_t size);
    void SetKernelArgumentLocal(const size_t size);
    uint64_t GetAttribute(const cl_kernel_work_group_info attribute) const;
};

} // namespace ktt

#endif // KTT_API_OPENCL
