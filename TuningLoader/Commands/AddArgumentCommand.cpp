#include <random>

#include <Commands/AddArgumentCommand.h>
#include <KttLoaderAssert.h>

namespace ktt
{

AddArgumentCommand::AddArgumentCommand(const ArgumentMemoryType memoryType, const ArgumentDataType dataType, const size_t size,
    const ArgumentAccessType accessType, const ArgumentFillType fillType, const float fillValue) :
    m_MemoryType(memoryType),
    m_Type(dataType),
    m_Size(size),
    m_AccessType(accessType),
    m_FillType(fillType),
    m_FillValue(fillValue)
{}

void AddArgumentCommand::Execute(TunerContext& context)
{
    if (m_MemoryType == ArgumentMemoryType::Scalar)
    {
        KttLoaderAssert(m_Type == ArgumentDataType::Int || m_Type == ArgumentDataType::Float, "Unsupported data type");
        ArgumentId id;

        if (m_Type == ArgumentDataType::Int)
        {
            id = context.GetTuner().AddArgumentScalar(static_cast<int>(m_FillValue));
        }
        else
        {
            id = context.GetTuner().AddArgumentScalar(m_FillValue);
        }
        
        context.GetArguments().push_back(id);
        context.GetTuner().SetArguments(context.GetKernelDefinitionId(), context.GetArguments());
        return;
    }

    KttLoaderAssert(m_MemoryType == ArgumentMemoryType::Vector, "Unsupported memory type");
    std::vector<float> input(m_Size);

    switch (m_FillType)
    {
    case ArgumentFillType::Constant:
        input.assign(m_Size, m_FillValue);
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
        KttLoaderError("Unhandled fill type");
    }

    const auto id = context.GetTuner().AddArgumentVector<float>(input, m_AccessType);
    context.GetArguments().push_back(id);
    context.GetTuner().SetArguments(context.GetKernelDefinitionId(), context.GetArguments());
}

CommandPriority AddArgumentCommand::GetPriority() const
{
    return CommandPriority::ArgumentAddition;
}

} // namespace ktt
