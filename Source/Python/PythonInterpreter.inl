#ifdef KTT_PYTHON

#include <Python/PythonInterpreter.h>

namespace ktt
{

template <typename T>
T PythonInterpreter::Evaluate(const std::string& expression)
{
    pybind11::gil_scoped_acquire acquire;
    pybind11::dict locals;
    return Evaluate<T>(expression, locals);
}

template <typename T>
T PythonInterpreter::Evaluate(const std::string& expression, pybind11::dict& locals)
{
    //gil_scoped_acquire does not need to be here, because it should be called in function that calls this, before putting the values in pybind11::dict& locals
    return pybind11::eval(expression, pybind11::globals(), locals).cast<T>();
}



} // namespace ktt

#endif // KTT_PYTHON
