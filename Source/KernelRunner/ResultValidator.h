#pragma once

#include <map>
#include <memory>
#include <type_traits>

#include <Kernel/Kernel.h>
#include <KernelRunner/KernelRunMode.h>
#include <KernelRunner/ValidationData.h>
#include <KernelRunner/ValidationMethod.h>
#include <KernelRunner/ValidationMode.h>
#include <KttTypes.h>

namespace ktt
{

class KernelRunner;

class ResultValidator
{
public:
    explicit ResultValidator(KernelRunner& kernelRunner);

    void SetValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    void SetValidationMode(const ValidationMode mode);

    void InitializeValidationData(const KernelArgument& argument);
    void SetValidationRange(const ArgumentId& id, const size_t range);
    void SetValueComparator(const ArgumentId& id, ValueComparator comparator);
    void SetReferenceComputation(const ArgumentId& id, ReferenceComputation computation);
    void SetReferenceKernel(const ArgumentId& id, const Kernel& kernel, const KernelConfiguration& configuration,
        const KernelDimensions& dimensions);
    void SetReferenceArgument(const ArgumentId& id, const KernelArgument& argument);
    bool HasValidationData(const ArgumentId& id) const;
    void RemoveValidationData(const ArgumentId& id);
    void RemoveDataWithReferenceKernel(const KernelId id);

    void ComputeReferenceResult(const Kernel& kernel, const KernelRunMode runMode);
    void ComputeReferenceResult(const ArgumentId& id);
    void ClearReferenceResult(const Kernel& kernel);
    void ClearReferenceResult(const ArgumentId& id);
    bool HasReferenceResult(const Kernel& kernel) const;
    bool HasReferenceResult(const ArgumentId& id) const;

    bool ValidateArguments(const Kernel& kernel, const KernelRunMode runMode) const;

private:
    KernelRunner& m_KernelRunner;
    double m_ToleranceThreshold;
    ValidationMethod m_ValidationMethod;
    ValidationMode m_ValidationMode;
    std::map<ArgumentId, std::unique_ptr<ValidationData>> m_ValidationData;

    bool IsRunModeValidated(const KernelRunMode mode) const;
    bool ValidateArgument(const KernelArgument& argument) const;
    bool ValidateResultWithMethod(const KernelArgument& argument, const void* result, const void* reference,
        const size_t range) const;
    bool ValidateResultWithComparator(const KernelArgument& argument, const void* result, const void* reference,
        const size_t range, ValueComparator comparator) const;

    template <typename T>
    bool ValidateResult(const KernelArgument& argument, const T* result, const T* reference, const size_t range) const;

    template <typename T>
    bool ValidateResultInner(const KernelArgument& argument, const T* result, const T* reference, const size_t range,
        std::true_type) const;

    template <typename T>
    bool ValidateResultInner(const KernelArgument& argument, const T* result, const T* reference, const size_t range,
        std::false_type) const;

    template <typename T>
    bool ValidateAbsoluteDifference(const KernelArgument& argument, const T* result, const T* reference, const size_t range) const;

    template <typename T>
    bool ValidateSideBySide(const KernelArgument& argument, const T* result, const T* reference, const size_t range) const;

    template <typename T>
    bool ValidateSideBySideRelative(const KernelArgument& argument, const T* result, const T* reference, const size_t range) const;
};

} // namespace ktt

#include <KernelRunner/ResultValidator.inl>
