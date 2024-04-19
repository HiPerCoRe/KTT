#ifdef KTT_PYTHON

#include <Python/PythonInterpreter.h>

namespace ktt
{

PythonInterpreter& PythonInterpreter::GetInterpreter()
{
    static PythonInterpreter interpreter;

    // lock the access to python interpreter and limit its use to a single thread
    interpreter.m_Mutex.lock();

    return interpreter;
}

void PythonInterpreter::ReleaseInterpreter()
{
    // unlock after use
    m_Mutex.unlock();
}

void PythonInterpreter::Execute(const std::string& script)
{
    pybind11::gil_scoped_acquire acquire;
    pybind11::dict locals;
    Execute(script, locals);
}

void PythonInterpreter::Execute(const std::string& script, pybind11::dict& locals)
{
    //gil_scoped_acquire does not need to be here, because it should be called in function that calls this, before putting the values in pybind11::dict& locals
    pybind11::exec(script, pybind11::globals(), locals);
}


} // namespace ktt

#endif // KTT_PYTHON
