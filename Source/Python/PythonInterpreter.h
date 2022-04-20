#pragma once

#ifdef KTT_PYTHON

#include <mutex>
#include <string>
#include <pybind11/embed.h>

#include <Utility/DisableCopyMove.h>

namespace ktt
{

class PythonInterpreter : public DisableCopyMove
{
public:
    static PythonInterpreter& GetInterpreter();

    bool Evaluate(const std::string& expression);
    bool Evaluate(const std::string& expression, pybind11::dict& locals);

    void Execute(const std::string& script);
    void Execute(const std::string& script, pybind11::dict& locals);

private:
    pybind11::scoped_interpreter m_Interpreter;
    std::mutex m_Mutex;

    PythonInterpreter() = default;
};

} // namespace ktt

#endif // KTT_PYTHON
