#pragma once

#include <map>
#include <string>

#include <Deserialization/SearcherType.h>
#include <TunerCommand.h>

namespace ktt
{

class SearcherCommand : public TunerCommand
{
public:
    SearcherCommand() = default;
    explicit SearcherCommand(const SearcherType type, const std::map<std::string, std::string>& attributes);

    virtual void Execute(TunerContext& context) override;
    virtual CommandPriority GetPriority() const override;

private:
    SearcherType m_Type;
    std::map<std::string, std::string> m_Attributes;
};

} // namespace ktt
