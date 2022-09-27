#ifdef KTT_PYTHON
#include <pybind11/stl.h>
#endif // KTT_PYTHON

#include <Api/KttException.h>
#include <Kernel/KernelConstraint/ScriptConstraint.h>
#include <Python/PythonInterpreter.h>
#include <Utility/Logger/Logger.h>

namespace ktt
{

ScriptConstraint::ScriptConstraint(const std::vector<const KernelParameter*>& parameters, const std::string& script) :
    KernelConstraint(parameters),
    m_Script(script)
{
#ifndef KTT_PYTHON
    throw KttException("Usage of script-based kernel constraints requires compilation of Python backend");
#else
    if (m_Script.empty())
    {
        throw KttException("Script constraint must be properly defined");
    }
#endif // KTT_PYTHON
}

bool ScriptConstraint::IsFulfilled([[maybe_unused]] const std::vector<const ParameterValue*>& values) const
{
#ifdef KTT_PYTHON
    auto& interpreter = PythonInterpreter::GetInterpreter();
    pybind11::dict locals;
    bool result = false;

    try
    {
        size_t valueIndex = 0;

        for (const auto& name : m_ParameterNames)
        {
            locals[name.c_str()] = *values[valueIndex];
            ++valueIndex;
        }

        result = interpreter.Evaluate<bool>(m_Script, locals);
    }
    catch (const pybind11::error_already_set& exception)
    {
        Logger::LogError(exception.what());
    }

    return result;
#else
    return false;
#endif // KTT_PYTHON
}

} // namespace ktt
