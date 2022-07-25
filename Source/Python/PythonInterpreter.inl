#ifdef KTT_PYTHON

#include <Python/PythonInterpreter.h>

namespace ktt
{

template <typename T>
T PythonInterpreter::Evaluate(const std::string& expression)
{
    pybind11::dict locals;
    return Evaluate<T>(expression, locals);
}

template <typename T>
T PythonInterpreter::Evaluate(const std::string& expression, pybind11::dict& locals)
{
    std::lock_guard<std::mutex> lock(m_Mutex);
    return pybind11::eval(expression, pybind11::globals(), locals).cast<T>();
}

} // namespace ktt

#endif // KTT_PYTHON
