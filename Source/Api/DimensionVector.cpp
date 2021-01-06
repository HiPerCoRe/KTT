#include <Api/DimensionVector.h>
#include <Utility/ErrorHandling/Assert.h>

namespace ktt
{

DimensionVector::DimensionVector() :
    DimensionVector(1, 1, 1)
{}

DimensionVector::DimensionVector(const size_t sizeX) :
    DimensionVector(sizeX, 1, 1)
{}

DimensionVector::DimensionVector(const size_t sizeX, const size_t sizeY) :
    DimensionVector(sizeX, sizeY, 1)
{}

DimensionVector::DimensionVector(const size_t sizeX, const size_t sizeY, const size_t sizeZ) :
    m_SizeX(sizeX),
    m_SizeY(sizeY),
    m_SizeZ(sizeZ)
{}

DimensionVector::DimensionVector(const std::vector<size_t>& vector) :
    DimensionVector(1, 1, 1)
{
    if (vector.size() > 0)
    {
        m_SizeX = vector[0];
    }
    if (vector.size() > 1)
    {
        m_SizeY = vector[1];
    }
    if (vector.size() > 2)
    {
        m_SizeZ = vector[2];
    }
}

void DimensionVector::SetSizeX(const size_t sizeX)
{
    m_SizeX = sizeX;
}

void DimensionVector::SetSizeY(const size_t sizeY)
{
    m_SizeY = sizeY;
}

void DimensionVector::SetSizeZ(const size_t sizeZ)
{
    m_SizeZ = sizeZ;
}

void DimensionVector::SetSize(const ModifierDimension modifierDimension, const size_t size)
{
    switch (modifierDimension)
    {
    case ModifierDimension::X:
        SetSizeX(size);
        break;
    case ModifierDimension::Y:
        SetSizeY(size);
        break;
    case ModifierDimension::Z:
        SetSizeZ(size);
        break;
    default:
        KttError("Unhandled modifier dimension value");
    }
}

void DimensionVector::Multiply(const DimensionVector& factor)
{
    m_SizeX *= factor.m_SizeX;
    m_SizeY *= factor.m_SizeY;
    m_SizeZ *= factor.m_SizeZ;
}

void DimensionVector::Divide(const DimensionVector& divisor)
{
    m_SizeX /= divisor.m_SizeX;
    m_SizeY /= divisor.m_SizeY;
    m_SizeZ /= divisor.m_SizeZ;
}

void DimensionVector::ModifyByValue(const size_t value, const ModifierAction modifierAction, const ModifierDimension modifierDimension)
{
    switch (modifierAction)
    {
    case ModifierAction::Add:
        AddValue(value, modifierDimension);
        break;
    case ModifierAction::Subtract:
        SubtractValue(value, modifierDimension);
        break;
    case ModifierAction::Multiply:
        MultiplyByValue(value, modifierDimension);
        break;
    case ModifierAction::Divide:
        DivideByValue(value, modifierDimension);
        break;
    case ModifierAction::DivideCeil:
        DivideCeilByValue(value, modifierDimension);
        break;
    default:
        KttError("Unhandled modifier action value");
    }
}

size_t DimensionVector::GetSizeX() const
{
    return m_SizeX;
}

size_t DimensionVector::GetSizeY() const
{
    return m_SizeY;
}

size_t DimensionVector::GetSizeZ() const
{
    return m_SizeZ;
}

size_t DimensionVector::GetSize(const ModifierDimension modifierDimension) const
{
    switch (modifierDimension)
    {
    case ModifierDimension::X:
        return m_SizeX;
    case ModifierDimension::Y:
        return m_SizeY;
    case ModifierDimension::Z:
        return m_SizeZ;
    default:
        KttError("Unhandled modifier dimension value");
        return 0;
    }
}

size_t DimensionVector::GetTotalSize() const
{
    return m_SizeX * m_SizeY * m_SizeZ;
}

std::vector<size_t> DimensionVector::GetVector() const
{
    std::vector<size_t> vector;
    vector.push_back(m_SizeX);
    vector.push_back(m_SizeY);
    vector.push_back(m_SizeZ);
    return vector;
}

std::string DimensionVector::GetString() const
{
    const std::string result = "(" + std::to_string(GetSizeX()) + ", " + std::to_string(GetSizeY()) + ", "
        + std::to_string(GetSizeZ()) + ")";
    return result;
}

bool DimensionVector::operator==(const DimensionVector& other) const
{
    return m_SizeX == other.m_SizeX && m_SizeY == other.m_SizeY && m_SizeZ == other.m_SizeZ;
}

bool DimensionVector::operator!=(const DimensionVector& other) const
{
    return !(*this == other);
}

void DimensionVector::AddValue(const size_t value, const ModifierDimension modifierDimension)
{
    switch (modifierDimension)
    {
    case ModifierDimension::X:
        m_SizeX += value;
        break;
    case ModifierDimension::Y:
        m_SizeY += value;
        break;
    case ModifierDimension::Z:
        m_SizeZ += value;
        break;
    default:
        KttError("Unhandled modifier dimension value");
    }
}

void DimensionVector::SubtractValue(const size_t value, const ModifierDimension modifierDimension)
{
    switch (modifierDimension)
    {
    case ModifierDimension::X:
        m_SizeX -= value;
        break;
    case ModifierDimension::Y:
        m_SizeY -= value;
        break;
    case ModifierDimension::Z:
        m_SizeZ -= value;
        break;
    default:
        KttError("Unhandled modifier dimension value");
    }
}

void DimensionVector::MultiplyByValue(const size_t value, const ModifierDimension modifierDimension)
{
    switch (modifierDimension)
    {
    case ModifierDimension::X:
        m_SizeX *= value;
        break;
    case ModifierDimension::Y:
        m_SizeY *= value;
        break;
    case ModifierDimension::Z:
        m_SizeZ *= value;
        break;
    default:
        KttError("Unhandled modifier dimension value");
    }
}

void DimensionVector::DivideByValue(const size_t value, const ModifierDimension modifierDimension)
{
    switch (modifierDimension)
    {
    case ModifierDimension::X:
        m_SizeX /= value;
        break;
    case ModifierDimension::Y:
        m_SizeY /= value;
        break;
    case ModifierDimension::Z:
        m_SizeZ /= value;
        break;
    default:
        KttError("Unhandled modifier dimension value");
    }
}

void DimensionVector::DivideCeilByValue(const size_t value, const ModifierDimension modifierDimension)
{
    switch (modifierDimension)
    {
    case ModifierDimension::X:
        m_SizeX = (m_SizeX + value - 1) / value;
        break;
    case ModifierDimension::Y:
        m_SizeY = (m_SizeY + value - 1) / value;
        break;
    case ModifierDimension::Z:
        m_SizeZ = (m_SizeZ + value - 1) / value;
        break;
    default:
        KttError("Unhandled modifier dimension value");
    }
}

} // namespace ktt
