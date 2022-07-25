#pragma once

#ifdef KTT_PYTHON

#include <mutex>
#include <string>
#include <pybind11/embed.h>

#include <Utility/DisableCopyMove.h>
#include <KttPlatform.h>

namespace ktt
{

class KTT_VISIBILITY_HIDDEN PythonInterpreter : public DisableCopyMove
{
public:
    static PythonInterpreter& GetInterpreter();

    template <typename T>
    T Evaluate(const std::string& expression);
    template <typename T>
    T Evaluate(const std::string& expression, pybind11::dict& locals);

    void Execute(const std::string& script);
    void Execute(const std::string& script, pybind11::dict& locals);

private:
    pybind11::scoped_interpreter m_Interpreter;
    std::mutex m_Mutex;

    PythonInterpreter() = default;
};

} // namespace ktt

#include <Python/PythonInterpreter.inl>

#endif // KTT_PYTHON
