#ifdef KTT_PYTHON
#include <pybind11/stl.h>
#endif // KTT_PYTHON

#include <Kernel/ThreadModifier.h>
#include <Python/PythonInterpreter.h>
#include <Utility/Logger/Logger.h>
#include <Utility/StlHelpers.h>

namespace ktt
{

ThreadModifier::ThreadModifier() :
    m_Function(nullptr)
{}

ThreadModifier::ThreadModifier(const std::vector<std::string>& parameters, const std::vector<KernelDefinitionId>& definitions,
    ModifierFunction function) :
    m_Parameters(parameters),
    m_Definitions(definitions),
    m_Function(function)
{}

ThreadModifier::ThreadModifier(const std::vector<KernelDefinitionId>& definitions, const std::string& script) :
    m_Definitions(definitions),
    m_Function(nullptr),
    m_Script(script)
{
#ifndef KTT_PYTHON
    throw KttException("Usage of script-based thread modifiers requires compilation of Python backend");
#endif // KTT_PYTHON
}

const std::vector<std::string>& ThreadModifier::GetParameters() const
{
    return m_Parameters;
}

const std::vector<KernelDefinitionId>& ThreadModifier::GetDefinitions() const
{
    return m_Definitions;
}

uint64_t ThreadModifier::GetModifiedSize(const KernelDefinitionId id, const uint64_t originalSize,
    const std::vector<ParameterPair>& pairs) const
{
    if (!ContainsElement(m_Definitions, id))
    {
        return originalSize;
    }

    if (!m_Script.empty())
    {
        return EvaluateScript(originalSize, pairs);
    }

    std::vector<uint64_t> values = ParameterPair::GetParameterValues<uint64_t>(pairs, m_Parameters);
    return m_Function(originalSize, values);
}

uint64_t ThreadModifier::EvaluateScript([[maybe_unused]] const uint64_t originalSize,
    [[maybe_unused]] const std::vector<ParameterPair>& pairs) const
{
#ifdef KTT_PYTHON
    auto& interpreter = PythonInterpreter::GetInterpreter();
    pybind11::dict locals;
    uint64_t result = 1;

    try
    {
        locals["defaultSize"] = originalSize;

        for (const auto& pair : pairs)
        {
            locals[pair.GetName().c_str()] = pair.GetValue();
        }

        result = interpreter.Evaluate<uint64_t>(m_Script, locals);
    }
    catch (const pybind11::error_already_set& exception)
    {
        Logger::LogError(exception.what());
    }

    return result;
#else
    return 1;
#endif // KTT_PYTHON
}

} // namespace ktt
