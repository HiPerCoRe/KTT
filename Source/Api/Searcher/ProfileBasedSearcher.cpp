#include <Api/Searcher/ProfileBasedSearcher.h>
#include <TuningRunner/ConfigurationData.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h> 

namespace py = pybind11;

namespace ktt
{

ProfileBasedSearcher::ProfileBasedSearcher() :
    Searcher()
{}

/*void ProfileBasedSearcher::Initialize(const ConfigurationData& data)
{
    py::scoped_interpreter guard{}; //FIXME

    py::object o(py::module_::import("profileSearcher").attr("PyProfilingSearcher"));
    o.attr("Initialize")(o());
    o.attr("OnInitialize")(o());

    OnInitialize();
}*/

void ProfileBasedSearcher::OnInitialize()
{
    py::scoped_interpreter guard{}; //FIXME

    m_PythonImplementation = new py::object(py::module_::import("profileSearcher").attr("PyProfilingSearcher"));
    ((py::object*)m_PythonImplementation)->attr("Initialize")((*((py::object*)m_PythonImplementation))(), m_Data);
    //((py::object*)m_PythonImplementation)->attr("OnInitialize")((*((py::object*)m_PythonImplementation)))();

    /*py::object o(py::module_::import("profileSearcher").attr("PyProfilingSearcher"));
    o.attr("Initialize")(o(), m_Data);
    o.attr("OnInitialize")(o());*/


    m_CurrentConfiguration = GetRandomConfiguration();
}

bool ProfileBasedSearcher::CalculateNextConfiguration([[maybe_unused]] const KernelResult& previousResult)
{
    m_CurrentConfiguration = GetRandomConfiguration();
    return true;
}

KernelConfiguration ProfileBasedSearcher::GetCurrentConfiguration() const
{
    return m_CurrentConfiguration;
}

} // namespace ktt
