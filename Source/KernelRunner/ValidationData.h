#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Api/Configuration/KernelConfiguration.h>
#include <KttTypes.h>

namespace ktt
{

class Kernel;
class KernelArgument;
class KernelRunner;

class ValidationData
{
public:
    explicit ValidationData(KernelRunner& kernelRunner, const KernelArgument& argument);

    void SetValidationRange(const size_t range);
    void SetValueComparator(ValueComparator comparator);
    void SetReferenceComputation(ReferenceComputation computation);
    void SetReferenceKernel(const Kernel& kernel, const KernelConfiguration& configuration, const KernelDimensions& dimensions);
    void SetReferenceArgument(const KernelArgument& argument);

    size_t GetValidationRange() const;
    ValueComparator GetValueComparator() const;
    bool HasValueComparator() const;
    bool HasReferenceComputation() const;
    bool HasReferenceKernel() const;
    bool HasReferenceArgument() const;
    KernelId GetReferenceKernelId() const;

    void ComputeReferenceResults();
    void ClearReferenceResults();
    bool HasReferenceResults() const;

    template <typename T>
    const T* GetReferenceResult() const;

private:
    KernelRunner& m_KernelRunner;
    const KernelArgument& m_Argument;
    size_t m_ValidationRange;
    ValueComparator m_Comparator;
    ReferenceComputation m_ReferenceComputation;
    std::vector<uint8_t> m_ReferenceResult;
    const Kernel* m_ReferenceKernel;
    KernelConfiguration m_ReferenceConfiguration;
    KernelDimensions m_ReferenceDimensions;
    std::vector<uint8_t> m_ReferenceKernelResult;
    const KernelArgument* m_ReferenceArgument;

    void ComputeReferenceWithFunction();
    void ComputeReferenceWithKernel();
    void ResetReferenceData();

    template <typename T>
    const T* GetReferenceComputationResult() const;

    template <typename T>
    const T* GetReferenceKernelResult() const;

    template <typename T>
    const T* GetReferenceArgumentResult() const;
};

} // namespace ktt

#include <KernelRunner/ValidationData.inl>
