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
    void ReleaseInterpreter();

    template <typename T>
    T Evaluate(const std::string& expression);
    template <typename T>
    T Evaluate(const std::string& expression, pybind11::dict& locals);

    void Execute(const std::string& script);
    void Execute(const std::string& script, pybind11::dict& locals);

private:
    pybind11::scoped_interpreter m_Interpreter{};
    std::mutex m_Mutex;
    std::unique_ptr<pybind11::gil_scoped_release> mp_gil_release;


    PythonInterpreter() {
      // global interpreter lock of Python interpreter needs to be released, so that we can manage when to acquire it later, when needed
      mp_gil_release = std::make_unique<pybind11::gil_scoped_release>();
    }
};

} // namespace ktt

#include <Python/PythonInterpreter.inl>

#endif // KTT_PYTHON
