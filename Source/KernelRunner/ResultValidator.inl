#include <cmath>

#include <KernelRunner/ResultValidator.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

template <typename T>
bool ResultValidator::ValidateResult(const KernelArgument& argument, const T* result, const T* reference, const size_t range) const
{
    return ValidateResultInner(argument, result, reference, range, std::is_floating_point<T>());
}

template <typename T>
bool ResultValidator::ValidateResultInner(const KernelArgument& argument, const T* result, const T* reference, const size_t range,
    std::true_type) const
{
    switch (m_ValidationMethod)
    {
    case ValidationMethod::AbsoluteDifference:
        return ValidateAbsoluteDifference(argument, result, reference, range);
    case ValidationMethod::SideBySideComparison:
        return ValidateSideBySide(argument, result, reference, range);
    case ValidationMethod::SideBySideRelativeComparison:
        return ValidateSideBySideRelative(argument, result, reference, range);
    default:
        KttError("Unhandled validation method value");
        return false;
    }
}

template <typename T>
bool ResultValidator::ValidateResultInner(const KernelArgument& argument, const T* result, const T* reference, const size_t range,
    std::false_type) const
{
    for (size_t i = 0; i < range; ++i)
    {
        if (result[i] != reference[i])
        {
            Logger::LogWarning("Results differ for argument with id " + std::to_string(argument.GetId()) + " at index " + std::to_string(i)
                + ", reference value: " + std::to_string(reference[i]) + ", result value: " + std::to_string(result[i])
                + ", difference: " + std::to_string(std::fabs(result[i] - reference[i])));
            return false;
        }
    }
    return true;
}

template <typename T>
bool ResultValidator::ValidateAbsoluteDifference(const KernelArgument& argument, const T* result, const T* reference, const size_t range) const
{
    double difference = 0.0;
    const size_t validatedElements = range / argument.GetElementSize();

    for (size_t i = 0; i < validatedElements; ++i)
    {
        difference += std::fabs(result[i] - reference[i]);
    }

    if (difference > m_ToleranceThreshold)
    {
        Logger::LogWarning("Results differ for argument with id " + std::to_string(argument.GetId()) + ", absolute difference is "
            + std::to_string(difference));
        return false;
    }

    return true;
}

template <typename T>
bool ResultValidator::ValidateSideBySide(const KernelArgument& argument, const T* result, const T* reference, const size_t range) const
{
    const size_t validatedElements = range / argument.GetElementSize();

    for (size_t i = 0; i < validatedElements; ++i)
    {
        const T difference = std::fabs(result[i] - reference[i]);

        if (difference > m_ToleranceThreshold)
        {
            Logger::LogWarning("Results differ for argument with id " + std::to_string(argument.GetId()) + " at index " + std::to_string(i)
                + ", reference value: " + std::to_string(reference[i]) + ", result value: " + std::to_string(result[i])
                + ", difference: " + std::to_string(difference));
            return false;
        }
    }

    return true;
}

template <typename T>
bool ResultValidator::ValidateSideBySideRelative(const KernelArgument& argument, const T* result, const T* reference, const size_t range) const
{
    const size_t validatedElements = range / argument.GetElementSize();

    for (size_t i = 0; i < validatedElements; ++i)
    {
        const T difference = std::fabs(result[i] - reference[i]);
        const T relativeDifference = difference / reference[i];

        if (difference > 1e-4 && relativeDifference > m_ToleranceThreshold)
        {
            Logger::LogWarning("Results differ for argument with id " + std::to_string(argument.GetId()) + " at index " + std::to_string(i)
                + ", reference value: " + std::to_string(reference[i]) + ", result value: " + std::to_string(result[i])
                + ", relative difference: " + std::to_string(relativeDifference));
            return false;
        }
    }
    return true;
}

} // namespace ktt
