#include <string>

#include <Api/KttException.h>
#include <KernelRunner/KernelRunner.h>
#include <KernelRunner/ResultValidator.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/External/half.hpp>
#include <Utility/StlHelpers.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

using half_float::half;

ResultValidator::ResultValidator(KernelRunner& kernelRunner) :
    m_KernelRunner(kernelRunner),
    m_ToleranceThreshold(1e-4),
    m_ValidationMethod(ValidationMethod::SideBySideComparison),
    m_ValidationMode(ValidationMode::OfflineTuning | ValidationMode::OnlineTuning)
{}

void ResultValidator::SetValidationMethod(const ValidationMethod method, const double toleranceThreshold)
{
    if (toleranceThreshold < 0.0)
    {
        throw KttException("Tolerance threshold cannot be negative");
    }

    m_ValidationMethod = method;
    m_ToleranceThreshold = toleranceThreshold;
}

void ResultValidator::SetValidationMode(const ValidationMode mode)
{
    m_ValidationMode = mode;
}

void ResultValidator::InitializeValidationData(const KernelArgument& argument)
{
    m_ValidationData[argument.GetId()] = std::make_unique<ValidationData>(m_KernelRunner, argument);
}

void ResultValidator::SetValidationRange(const ArgumentId id, const size_t range)
{
    KttAssert(HasValidationData(id), "Validation data not found");
    m_ValidationData[id]->SetValidationRange(range);
}

void ResultValidator::SetValueComparator(const ArgumentId id, ValueComparator comparator)
{
    KttAssert(HasValidationData(id), "Validation data not found");
    m_ValidationData[id]->SetValueComparator(comparator);
}

void ResultValidator::SetReferenceComputation(const ArgumentId id, ReferenceComputation computation)
{
    KttAssert(HasValidationData(id), "Validation data not found");
    m_ValidationData[id]->SetReferenceComputation(computation);
}

void ResultValidator::SetReferenceKernel(const ArgumentId id, const Kernel& kernel, const KernelConfiguration& configuration)
{
    KttAssert(HasValidationData(id), "Validation data not found");
    m_ValidationData[id]->SetReferenceKernel(kernel, configuration);
}

bool ResultValidator::HasValidationData(const ArgumentId id) const
{
    return ContainsKey(m_ValidationData, id);
}

void ResultValidator::ComputeReferenceResult(const Kernel& kernel, const KernelRunMode runMode)
{
    if (!IsRunModeValidated(runMode))
    {
        return;
    }

    for (const auto* argument : kernel.GetVectorArguments())
    {
        const auto id = argument->GetId();

        if (HasValidationData(id))
        {
            ComputeReferenceResult(id);
        }
    }
}

void ResultValidator::ComputeReferenceResult(const ArgumentId id)
{
    KttAssert(HasValidationData(id), "Validation data not found");
    m_ValidationData[id]->ComputeReferenceResults();
}

void ResultValidator::ClearReferenceResult(const Kernel& kernel)
{
    for (const auto* argument : kernel.GetVectorArguments())
    {
        const auto id = argument->GetId();

        if (HasValidationData(id))
        {
            ClearReferenceResult(id);
        }
    }
}

void ResultValidator::ClearReferenceResult(const ArgumentId id)
{
    KttAssert(HasValidationData(id), "Validation data not found");
    m_ValidationData[id]->ClearReferenceResults();
}

bool ResultValidator::HasReferenceResult(const Kernel& kernel) const
{
    bool result = true;

    for (const auto* argument : kernel.GetVectorArguments())
    {
        const auto id = argument->GetId();

        if (HasValidationData(id))
        {
            result &= HasReferenceResult(id);
        }
    }

    return result;
}

bool ResultValidator::HasReferenceResult(const ArgumentId id) const
{
    KttAssert(HasValidationData(id), "Validation data not found");
    return m_ValidationData.find(id)->second->HasReferenceResults();
}

bool ResultValidator::ValidateArguments(const Kernel& kernel, const KernelRunMode runMode) const
{
    if (!IsRunModeValidated(runMode))
    {
        return true;
    }

    bool result = true;

    for (const auto* argument : kernel.GetVectorArguments())
    {
        const auto id = argument->GetId();

        if (HasValidationData(id))
        {
            result &= ValidateArgument(*argument);
        }
    }

    return result;
}

bool ResultValidator::IsRunModeValidated(const KernelRunMode mode) const
{
    switch (mode)
    {
    case KernelRunMode::Running:
        return HasFlag(m_ValidationMode, ValidationMode::Running);
    case KernelRunMode::OfflineTuning:
        return HasFlag(m_ValidationMode, ValidationMode::OfflineTuning);
    case KernelRunMode::OnlineTuning:
        return HasFlag(m_ValidationMode, ValidationMode::OnlineTuning);
    case KernelRunMode::ResultValidation:
        return false;
    default:
        KttError("Unhandled kernel run mode value");
        return false;
    }
}

bool ResultValidator::ValidateArgument(const KernelArgument& argument) const
{
    const auto& validationData = *m_ValidationData.find(argument.GetId())->second;
    const size_t validationRange = validationData.GetValidationRange();
    const size_t bufferSize = validationRange * argument.GetElementSize();

    std::vector<uint8_t> argumentData(bufferSize);
    BufferOutputDescriptor descriptor(argument.GetId(), argumentData.data(), bufferSize);
    m_KernelRunner.DownloadBuffers({descriptor});

    bool result = true;

    if (validationData.HasValueComparator())
    {
        if (validationData.HasReferenceComputation())
        {
            result &= ValidateResultWithComparator(argument, argumentData.data(), validationData.GetReferenceResult<void>(),
                validationRange, validationData.GetValueComparator());
        }

        if (validationData.HasReferenceKernel())
        {
            result &= ValidateResultWithComparator(argument, argumentData.data(), validationData.GetReferenceKernelResult<void>(),
                validationRange, validationData.GetValueComparator());
        }

        return result;
    }

    if (validationData.HasReferenceComputation())
    {
        result &= ValidateResultWithMethod(argument, argumentData.data(), validationData.GetReferenceResult<void>(),
            validationRange);
    }

    if (validationData.HasReferenceKernel())
    {
        result &= ValidateResultWithMethod(argument, argumentData.data(), validationData.GetReferenceKernelResult<void>(),
            validationRange);
    }

    return result;
}

bool ResultValidator::ValidateResultWithMethod(const KernelArgument& argument, const void* result, const void* reference,
    const size_t range) const
{
    switch (argument.GetDataType())
    {
    case ArgumentDataType::Char:
        return ValidateResult<int8_t>(argument, static_cast<const int8_t*>(result), static_cast<const int8_t*>(reference), range);
    case ArgumentDataType::UnsignedChar:
        return ValidateResult<uint8_t>(argument, static_cast<const uint8_t*>(result), static_cast<const uint8_t*>(reference), range);
    case ArgumentDataType::Short:
        return ValidateResult<int16_t>(argument, static_cast<const int16_t*>(result), static_cast<const int16_t*>(reference), range);
    case ArgumentDataType::UnsignedShort:
        return ValidateResult<uint16_t>(argument, static_cast<const uint16_t*>(result), static_cast<const uint16_t*>(reference), range);
    case ArgumentDataType::Int:
        return ValidateResult<int32_t>(argument, static_cast<const int32_t*>(result), static_cast<const int32_t*>(reference), range);
    case ArgumentDataType::UnsignedInt:
        return ValidateResult<uint32_t>(argument, static_cast<const uint32_t*>(result), static_cast<const uint32_t*>(reference), range);
    case ArgumentDataType::Long:
        return ValidateResult<int64_t>(argument, static_cast<const int64_t*>(result), static_cast<const int64_t*>(reference), range);
    case ArgumentDataType::UnsignedLong:
        return ValidateResult<uint64_t>(argument, static_cast<const uint64_t*>(result), static_cast<const uint64_t*>(reference), range);
    case ArgumentDataType::Half:
        return ValidateResult<half>(argument, static_cast<const half*>(result), static_cast<const half*>(reference), range);
    case ArgumentDataType::Float:
        return ValidateResult<float>(argument, static_cast<const float*>(result), static_cast<const float*>(reference), range);
    case ArgumentDataType::Double:
        return ValidateResult<double>(argument, static_cast<const double*>(result), static_cast<const double*>(reference), range);
    case ArgumentDataType::Custom:
        throw KttException("Validation of kernel arguments with custom data type requires usage of value comparator");
    default:
        KttError("Unhandled argument data type value");
        return false;
    }
}

bool ResultValidator::ValidateResultWithComparator(const KernelArgument& argument, const void* result, const void* referenceResult,
    const size_t range, ValueComparator comparator) const
{
    const size_t elementSize = argument.GetElementSize();
    const size_t bufferSize = range * elementSize;

    for (size_t i = 0; i < bufferSize; i += elementSize)
    {
        if (!comparator(reinterpret_cast<const uint8_t*>(result) + i, reinterpret_cast<const uint8_t*>(referenceResult) + i))
        {
            Logger::LogWarning("Results differ for argument with id: " + std::to_string(argument.GetId()) + " at index "
                + std::to_string(i / elementSize));
            return false;
        }
    }

    return true;
}

} // namespace ktt
