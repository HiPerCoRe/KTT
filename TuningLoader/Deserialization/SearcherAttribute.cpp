#include <Deserialization/SearcherAttribute.h>

namespace ktt
{

SearcherAttribute::SearcherAttribute(const std::string& name, const std::string& value) :
    m_Name(name),
    m_Value(value)
{}

std::pair<std::string, std::string> SearcherAttribute::GeneratePair() const
{
    return std::make_pair(m_Name, m_Value);
}

} // namespace ktt
