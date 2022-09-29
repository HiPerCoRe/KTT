#include <random>

#include <Commands/AddArgumentCommand.h>
#include <KttLoaderAssert.h>

namespace ktt
{

AddArgumentCommand::AddArgumentCommand(const ArgumentMemoryType memoryType, const ArgumentDataType dataType, const size_t size,
    const size_t typeSize, const ArgumentAccessType accessType, const ArgumentFillType fillType, const float fillValue) :
    m_MemoryType(memoryType),
    m_Type(dataType),
    m_Size(size),
    m_TypeSize(typeSize),
    m_AccessType(accessType),
    m_FillType(fillType),
    m_FillValue(fillValue)
{}

AddArgumentCommand::AddArgumentCommand(const ArgumentMemoryType memoryType, const ArgumentDataType dataType, const size_t size,
    const size_t typeSize, const ArgumentAccessType accessType, const ArgumentFillType fillType, const std::string& dataSource) :
    m_MemoryType(memoryType),
    m_Type(dataType),
    m_TypeSize(typeSize),
    m_Size(size),
    m_AccessType(accessType),
    m_FillType(fillType),
    m_FillValue(0.0f),
    m_DataSource(dataSource)
{}

void AddArgumentCommand::Execute(TunerContext& context)
{
    ArgumentId id = InvalidArgumentId;

    switch (m_MemoryType)
    {
    case ArgumentMemoryType::Scalar:
        id = SubmitScalarArgument(context);
        break;
    case ArgumentMemoryType::Vector:
        id = SubmitVectorArgument(context);
        break;
    case ArgumentMemoryType::Local:
        throw KttException("Unsupported memory type (local)");
    case ArgumentMemoryType::Symbol:
        throw KttException("Unsupported memory type (symbol)");
    default:
        KttLoaderError("Unhandled memory type");
        return;
    }

    context.GetArguments().push_back(id);
    context.GetTuner().SetArguments(context.GetKernelDefinitionId(), context.GetArguments());
}

CommandPriority AddArgumentCommand::GetPriority() const
{
    return CommandPriority::KernelArgument;
}

ArgumentId AddArgumentCommand::SubmitScalarArgument(TunerContext& context) const
{
    KttLoaderAssert(m_Type == ArgumentDataType::Int || m_Type == ArgumentDataType::Float, "Unsupported data type");

    if (m_Type == ArgumentDataType::Int)
    {
        return context.GetTuner().AddArgumentScalar(static_cast<int>(m_FillValue));
    }
    
    return context.GetTuner().AddArgumentScalar(m_FillValue);
}

ArgumentId AddArgumentCommand::SubmitVectorArgument(TunerContext& context) const
{
    switch (m_FillType)
    {
    case ArgumentFillType::Constant:
    {
        std::vector<float> input(m_Size);
        input.assign(m_Size, m_FillValue);
        return context.GetTuner().AddArgumentVector<float>(input, m_AccessType);
    }
    case ArgumentFillType::Random:
    {
        std::vector<float> input(m_Size);
        std::random_device device;
        std::default_random_engine engine(device());
        std::uniform_real_distribution<float> distribution(0.0f, m_FillValue);

        for (size_t i = 0; i < m_Size; ++i)
        {
            input[i] = distribution(engine);
        }

        return context.GetTuner().AddArgumentVector<float>(input, m_AccessType);
    }
    case ArgumentFillType::Generator:
    {
        return context.GetTuner().AddArgumentVectorFromGenerator(m_DataSource, m_Type, m_Size * m_TypeSize, m_TypeSize, m_AccessType);
    }
    case ArgumentFillType::BinaryRaw:
    {
        const auto path = context.GetFullPath(m_DataSource);
        return context.GetTuner().AddArgumentVectorFromFile(path, m_Type, m_TypeSize, m_AccessType);
    }
    default:
        KttLoaderError("Unhandled fill type");
        return InvalidArgumentId;
    }
}

} // namespace ktt
