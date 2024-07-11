#include <random>

#include <Commands/AddArgumentCommand.h>
#include <KttLoaderAssert.h>

namespace ktt
{
AddArgumentCommand::AddArgumentCommand(const ArgumentId& id, const ArgumentMemoryType memoryType, const ArgumentDataType dataType,
    const size_t elementCount, const size_t typeSize, const ArgumentAccessType accessType, const ArgumentFillType fillType,
    const float fillValue, const float randomSeed) :
    m_Id(id),
    m_MemoryType(memoryType),
    m_Type(dataType),
    m_ElementCount(elementCount),
    m_ElementSize(typeSize),
    m_AccessType(accessType),
    m_FillType(fillType),
    m_FillValue(fillValue),
    m_RandomSeed(randomSeed),
    m_IsReference(false)
{}

AddArgumentCommand::AddArgumentCommand(const ArgumentId& id, const ArgumentMemoryType memoryType, const ArgumentDataType dataType,
    const size_t elementCount, const size_t typeSize, const ArgumentAccessType accessType, const ArgumentFillType fillType,
    const std::string& dataSource) :
    m_Id(id),
    m_MemoryType(memoryType),
    m_Type(dataType),
    m_ElementCount(elementCount),
    m_ElementSize(typeSize),
    m_AccessType(accessType),
    m_FillType(fillType),
    m_FillValue(0.0f),
    m_DataSource(dataSource),
    m_RandomSeed(std::nanf("")),
    m_IsReference(false)
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

    if (!m_IsReference)
    {
        context.GetArguments().push_back(id);
        context.GetTuner().SetArguments(context.GetKernelDefinitionId(), context.GetArguments());
    }
}

CommandPriority AddArgumentCommand::GetPriority() const
{
    return CommandPriority::KernelArgument;
}

void AddArgumentCommand::SetReferenceProperties(const AddArgumentCommand& other)
{
    m_MemoryType = other.m_MemoryType;
    m_Type = other.m_Type;
    m_ElementCount = other.m_ElementCount;
    m_ElementSize = other.m_ElementSize;
    m_AccessType = other.m_AccessType;
    m_IsReference = true;
}

const ArgumentId& AddArgumentCommand::GetId() const
{
    return m_Id;
}

ArgumentId AddArgumentCommand::SubmitScalarArgument(TunerContext& context) const
{
    KttLoaderAssert(m_Type == ArgumentDataType::Int || m_Type == ArgumentDataType::Float, "Unsupported data type");

    if (m_Type == ArgumentDataType::Int)
    {
        return context.GetTuner().AddArgumentScalar(static_cast<int>(m_FillValue), m_Id);
    }
    
    return context.GetTuner().AddArgumentScalar(m_FillValue, m_Id);
}

ArgumentId AddArgumentCommand::SubmitVectorArgument(TunerContext& context) const
{
    switch (m_FillType)
    {
    case ArgumentFillType::Constant:
    {
        std::vector<float> input(m_ElementCount);
        input.assign(m_ElementCount, m_FillValue);
        return context.GetTuner().AddArgumentVector<float>(input, m_AccessType, m_Id);
    }
    case ArgumentFillType::Random:
    {
        std::vector<float> input(m_ElementCount);
        std::random_device device;
        std::default_random_engine engine;
        if (std::isnan(m_RandomSeed))
            engine = std::default_random_engine(device());
        else
            engine = std::default_random_engine(m_RandomSeed);
        std::uniform_real_distribution<float> distribution(0.0f, m_FillValue);

        for (size_t i = 0; i < m_ElementCount; ++i)
        {
            input[i] = distribution(engine);
        }

        return context.GetTuner().AddArgumentVector<float>(input, m_AccessType, m_Id);
    }
    case ArgumentFillType::Generator:
    {
        return context.GetTuner().AddArgumentVectorFromGenerator(m_DataSource, m_Type, m_ElementCount * m_ElementSize, m_ElementSize,
            m_AccessType, ArgumentMemoryLocation::Device, ArgumentManagementType::Framework, m_Id);
    }
    case ArgumentFillType::BinaryRaw:
    {
        const auto path = context.GetFullPath(m_DataSource);
        return context.GetTuner().AddArgumentVectorFromFile(path, m_Type, m_ElementSize, m_AccessType, ArgumentMemoryLocation::Device,
            ArgumentManagementType::Framework, m_Id);
    }
    default:
        KttLoaderError("Unhandled fill type");
        return InvalidArgumentId;
    }
}

} // namespace ktt
