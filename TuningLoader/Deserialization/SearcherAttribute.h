#pragma once

#include <string>
#include <utility>

namespace ktt
{

class SearcherAttribute
{
public:
    SearcherAttribute() = default;
    explicit SearcherAttribute(const std::string& name, const std::string& value);

    std::pair<std::string, std::string> GeneratePair() const;

private:
    std::string m_Name;
    std::string m_Value;
};

} // namespace ktt
