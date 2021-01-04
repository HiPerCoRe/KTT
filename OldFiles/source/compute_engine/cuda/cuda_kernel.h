#pragma once

#include <string>
#include <cuda.h>
#include <api/kernel_compilation_data.h>
#include <compute_engine/cuda/cuda_utility.h>

namespace ktt
{

class CUDAKernel
{
public:
    explicit CUDAKernel(const std::string& ptxSource, const std::string& kernelName) :
        kernelName(kernelName)
    {
        checkCUDAError(cuModuleLoadDataEx(&module, &ptxSource[0], 0, nullptr, nullptr), "cuModuleLoadDataEx");
        checkCUDAError(cuModuleGetFunction(&kernel, module, &kernelName[0]), "cuModuleGetFunction");
    }

    ~CUDAKernel()
    {
        checkCUDAError(cuModuleUnload(module), "cuModuleUnload");
    }

    CUmodule getModule() const
    {
        return module;
    }

    const std::string& getKernelName() const
    {
        return kernelName;
    }

    CUfunction getKernel() const
    {
        return kernel;
    }

    KernelCompilationData getCompilationData() const
    {
        KernelCompilationData result;

        collectAttribute(result.maxWorkGroupSize, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        collectAttribute(result.localMemorySize, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
        collectAttribute(result.privateMemorySize, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);
        collectAttribute(result.constantMemorySize, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES);
        collectAttribute(result.registersCount, CU_FUNC_ATTRIBUTE_NUM_REGS);

        return result;
    }

private:
    CUmodule module;
    std::string kernelName;
    CUfunction kernel;

    void collectAttribute(uint64_t& output, const CUfunction_attribute attribute) const
    {
        int attributeValue;
        checkCUDAError(cuFuncGetAttribute(&attributeValue, attribute, kernel), "cuFuncGetAttribute");
        output = static_cast<uint64_t>(attributeValue);
    }
};

} // namespace ktt
