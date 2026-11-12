#include <cstdint>
#include <optional>

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

    std::optional<uint64_t> seed;

    if (m_Attributes.count("seed") > 0)
    {
        seed = std::stoull(m_Attributes["seed"]);
    }

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
        {
          // if default values needs to be changed, do it also in Source/Tuner.cpp
          uint batchSize = 5;
          if (m_Attributes.count("batchSize") > 0)
            batchSize = std::stoul(m_Attributes["batchSize"]);
          uint neighborSize = 100;
          if (m_Attributes.count("neighborSize") > 0)
            neighborSize = std::stoul(m_Attributes["neighborSize"]);
          uint randomSize = 10;
          if (m_Attributes.count("randomSize") > 0)
            randomSize = std::stoul(m_Attributes["randomSize"]);
          context.GetTuner().SetProfileBasedSearcher(context.GetKernelId(), m_Attributes["modelPath"], true, batchSize, neighborSize, randomSize, seed);
          return;
        }
    default:
        KttLoaderError("Unhandled searcher type");
    }

    if (seed.has_value())
    {
        searcher->SetSeed(seed.value());
    }

    context.GetTuner().SetSearcher(context.GetKernelId(), std::move(searcher));
}

CommandPriority SearcherCommand::GetPriority() const
{
    return CommandPriority::Searcher;
}

} // namespace ktt
