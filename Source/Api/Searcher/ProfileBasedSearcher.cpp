#include <Api/Searcher/ProfileBasedSearcher.h>
//#include <TuningRunner/ConfigurationData.h>
#include <Ktt.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h> 

namespace py = pybind11;

namespace ktt
{

ProfileBasedSearcher::ProfileBasedSearcher(void* tuner, KernelId kernel, std::string model) :
    Searcher()
{
    py::scoped_interpreter guard{};
    m_PythonImplementation = new py::module_(py::module_::import("ProfileSearcherExecutor"));
    m_PythonImplementation = new py::object(((py::module_*)m_PythonImplementation)->attr("executeSearcher")(&*((Tuner*)tuner), kernel, model));
}

void ProfileBasedSearcher::OnInitialize()
{
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
