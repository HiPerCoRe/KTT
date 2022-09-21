#include <Commands/SearcherCommand.h>
#include <KttLoaderAssert.h>

namespace ktt
{

SearcherCommand::SearcherCommand(const SearcherType type, const std::map<std::string, std::string>& attributes) :
    m_Type(type),
    m_Attributes(attributes)
{}

void SearcherCommand::Execute(TunerContext& context)
{
    std::unique_ptr<Searcher> searcher;

    switch (m_Type)
    {
    case SearcherType::Deterministic:
        searcher = std::make_unique<DeterministicSearcher>();
        break;
    case SearcherType::Random:
        searcher = std::make_unique<RandomSearcher>();
        break;
    case SearcherType::MCMC:
        searcher = std::make_unique<McmcSearcher>();
        break;
    case SearcherType::ProfileBased:
        context.GetTuner().SetProfileBasedSearcher(context.GetKernelId(), m_Attributes["modelPath"]);
        return;
    default:
        KttLoaderError("Unhandled searcher type");
    }

    context.GetTuner().SetSearcher(context.GetKernelId(), std::move(searcher));
}

CommandPriority SearcherCommand::GetPriority() const
{
    return CommandPriority::Searcher;
}

} // namespace ktt
