#include <algorithm>

#include <Api/StopCondition/UnionCondition.h>

namespace ktt
{

UnionCondition::UnionCondition(const std::vector<std::shared_ptr<StopCondition>>& conditions) :
    m_Conditions(conditions)
{}

bool UnionCondition::IsFulfilled() const
{
    const bool fulfilled = std::any_of(m_Conditions.cbegin(), m_Conditions.cend(), [](const auto& condition)
    {
        return condition->IsFulfilled();
    });

    return fulfilled;
}

void UnionCondition::Initialize(const uint64_t configurationsCount)
{
    for (auto& condition : m_Conditions)
    {
        condition->Initialize(configurationsCount);
    }
}

void UnionCondition::Update(const KernelResult& result)
{
    for (auto& condition : m_Conditions)
    {
        condition->Update(result);
    }
}

std::string UnionCondition::GetStatusString() const
{
    std::string finalResult;

    for (auto& condition : m_Conditions)
    {
        finalResult += condition->GetStatusString();

        if (&condition != &m_Conditions.back())
        {
            finalResult += "\n";
        }
    }

    return finalResult;
}

} // namespace ktt
