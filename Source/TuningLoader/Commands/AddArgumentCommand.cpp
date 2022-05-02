#include <random>

#include <TuningLoader/Commands/AddArgumentCommand.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

AddArgumentCommand::AddArgumentCommand(const ArgumentDataType dataType, const size_t size, const ArgumentAccessType accessType,
    const ArgumentFillType fillType, const float fillValue, const size_t order) :
    m_Type(dataType),
    m_Size(size),
    m_AccessType(accessType),
    m_FillType(fillType),
    m_FillValue(fillValue),
    m_Order(order)
{}

void AddArgumentCommand::Execute(TunerContext& context)
{
    std::vector<float> input(m_Size);

    switch (m_FillType)
    {
    case ArgumentFillType::Constant:
        input.assign(m_Size, m_FillValue);
        break;
    case ArgumentFillType::Ascending:
        for (size_t i = 0; i < m_Size; ++i)
        {
            auto value = m_FillValue;
            input[i] = value;
            ++value;
        }
        break;
    case ArgumentFillType::Random:
    {
        std::random_device device;
        std::default_random_engine engine(device());
        std::uniform_real_distribution<float> distribution(0.0f, m_FillValue);

        for (size_t i = 0; i < m_Size; ++i)
        {
            input[i] = distribution(engine);
        }
        break;
    }
    default:
        KttError("Unhandled fill type");
    }

    const auto id = context.GetTuner().AddArgumentVector<float>(input, m_AccessType);
    context.GetArguments().push_back(id);
    context.GetTuner().SetArguments(context.GetKernelDefinitionId(), context.GetArguments());
}

CommandPriority AddArgumentCommand::GetPriority() const
{
    return CommandPriority::ArgumentAddition;
}

size_t AddArgumentCommand::GetOrder() const
{
    return m_Order;
}

} // namespace ktt
