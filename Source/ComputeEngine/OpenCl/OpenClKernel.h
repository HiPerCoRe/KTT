#pragma once

#ifdef KTT_API_OPENCL

#include <memory>
#include <string>
#include <CL/cl.h>

#include <Api/Configuration/DimensionVector.h>
#include <ComputeEngine/OpenCl/OpenClProgram.h>
#include <ComputeEngine/GlobalSizeType.h>
#include <Utility/IdGenerator.h>
#include <KttTypes.h>

namespace ktt
{

class EngineConfiguration;
class KernelArgument;
class OpenClBuffer;
class OpenClCommandQueue;
class OpenClComputeAction;
struct KernelCompilationData;

class OpenClKernel : public std::enable_shared_from_this<OpenClKernel>
{
public:
    explicit OpenClKernel(std::unique_ptr<OpenClProgram> program, const std::string& name,
        IdGenerator<ComputeActionId>& generator, const EngineConfiguration& configuration);
    ~OpenClKernel();

    std::unique_ptr<OpenClComputeAction> Launch(const OpenClCommandQueue& queue, const DimensionVector& globalSize,
        const DimensionVector& localSize);
    void SetArgument(const KernelArgument& argument);
    void SetArgument(OpenClBuffer& buffer);
    void ResetArguments();
    std::unique_ptr<KernelCompilationData> GenerateCompilationData() const;

    const std::string& GetName() const;
    cl_kernel GetKernel() const;
    uint64_t GetAttribute(const cl_kernel_work_group_info attribute) const;

private:
    std::string m_Name;
    std::unique_ptr<OpenClProgram> m_Program;
    IdGenerator<ComputeActionId>& m_Generator;
    const EngineConfiguration& m_Configuration;
    cl_kernel m_Kernel;
    cl_uint m_NextArgumentIndex;

    void SetKernelArgumentVector(const void* buffer);
    void SetKernelArgumentVectorSvm(const void* buffer);
    void SetKernelArgumentScalar(const void* value, const size_t size);
    void SetKernelArgumentLocal(const size_t size);
    DimensionVector AdjustGlobalSize(const DimensionVector& globalSize, const DimensionVector& localSize) const;
};

} // namespace ktt

#endif // KTT_API_OPENCL
