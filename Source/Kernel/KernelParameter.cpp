#ifdef KTT_PYTHON
#include <pybind11/stl.h>
#endif // KTT_PYTHON

#include <Api/KttException.h>
#include <Kernel/KernelParameter.h>
#include <Python/PythonInterpreter.h>
#include <Utility/ErrorHandling/Assert.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

static const std::string DefaultGroup = "KTTDefaultGroup";

KernelParameter::KernelParameter(const std::string& name, const std::vector<ParameterValue>& values, const std::string& group) :
    m_Name(name),
    m_Group(group),
    m_Values(values)
{
    if (values.empty())
    {
        throw KttException("Kernel parameter must have at least one value defined");
    }

    if (group.empty())
    {
        m_Group = DefaultGroup;
    }
}

KernelParameter::KernelParameter(const std::string& name, const ParameterValueType valueType, const std::string& valueScript,
    const std::string& group) :
    KernelParameter(name, GetValuesFromScript(valueType, valueScript), group)
{}

const std::string& KernelParameter::GetName() const
{
    return m_Name;
}

const std::string& KernelParameter::GetGroup() const
{
    return m_Group;
}

size_t KernelParameter::GetValuesCount() const
{
    return m_Values.size();
}

const std::vector<ParameterValue>& KernelParameter::GetValues() const
{
    return m_Values;
}

ParameterValueType KernelParameter::GetValueType() const
{
    return ParameterPair::GetTypeFromValue(m_Values[0]);
}

ParameterPair KernelParameter::GeneratePair(const size_t valueIndex) const
{
    if (valueIndex >= GetValuesCount())
    {
        throw KttException("Parameter value index is out of range");
    }

    return ParameterPair(m_Name, m_Values[valueIndex]);
}

std::vector<ParameterPair> KernelParameter::GeneratePairs() const
{
    std::vector<ParameterPair> result;

    for (size_t i = 0; i < GetValuesCount(); ++i)
    {
        result.push_back(GeneratePair(i));
    }

    return result;
}

bool KernelParameter::operator==(const KernelParameter& other) const
{
    return m_Name == other.m_Name;
}

bool KernelParameter::operator!=(const KernelParameter& other) const
{
    return !(*this == other);
}

bool KernelParameter::operator<(const KernelParameter& other) const
{
    return m_Name < other.m_Name;
}

std::vector<ParameterValue> KernelParameter::GetValuesFromScript([[maybe_unused]] const ParameterValueType valueType,
    [[maybe_unused]] const std::string& valueScript)
{
#ifndef KTT_PYTHON
    throw KttException("Usage of script-based kernel parameters requires compilation of Python backend");
#else
    auto& interpreter = PythonInterpreter::GetInterpreter();
    pybind11::gil_scoped_acquire acquire;
    std::vector<ParameterValue> result;

    try
    {
        switch (valueType)
        {
        case ParameterValueType::Int:
        {
            const auto values = interpreter.Evaluate<std::vector<int64_t>>(valueScript);

            for (const auto value : values)
            {
                result.push_back(value);
            }

            break;
        }
        case ParameterValueType::UnsignedInt:
        {
            const auto values = interpreter.Evaluate<std::vector<uint64_t>>(valueScript);

            for (const auto value : values)
            {
                result.push_back(value);
            }

            break;
        }
        case ParameterValueType::Double:
        {
            const auto values = interpreter.Evaluate<std::vector<double>>(valueScript);

            for (const auto value : values)
            {
                result.push_back(value);
            }

            break;
        }
        case ParameterValueType::Bool:
        {
            const auto values = interpreter.Evaluate<std::vector<bool>>(valueScript);

            for (const auto value : values)
            {
                result.push_back(value);
            }

            break;
        }
        case ParameterValueType::String:
        {
            const auto values = interpreter.Evaluate<std::vector<std::string>>(valueScript);

            for (const auto& value : values)
            {
                result.push_back(value);
            }

            break;
        }
        default:
            KttError("Unhandled parameter value type");
        }
    }
    catch (const pybind11::error_already_set& exception)
    {
        Logger::LogError(exception.what());
        interpreter.ReleaseInterpreter();
    }
    catch (const std::exception &e) {
        Logger::LogError(e.what());
        interpreter.ReleaseInterpreter();
    }
    interpreter.ReleaseInterpreter();

    return result;
#endif // KTT_PYTHON
}

} // namespace ktt
