#ifdef KTT_PYTHON

#include <Python/PythonInterpreter.h>

namespace ktt
{

PythonInterpreter& PythonInterpreter::GetInterpreter()
{
    static PythonInterpreter interpreter;
    return interpreter;
}

void PythonInterpreter::Execute(const std::string& script)
{
    pybind11::dict locals;
    Execute(script, locals);
}

void PythonInterpreter::Execute(const std::string& script, pybind11::dict& locals)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    pybind11::exec(script, pybind11::globals(), locals);
}

} // namespace ktt

#endif // KTT_PYTHON
