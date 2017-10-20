#include <stdexcept>
#include "dimension_vector.h"

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
    sizeX(sizeX),
    sizeY(sizeY),
    sizeZ(sizeZ)
{}

DimensionVector::DimensionVector(const std::vector<size_t>& vector) :
    DimensionVector(1, 1, 1)
{
    if (vector.size() > 0)
    {
        sizeX = vector.at(0);
    }
    if (vector.size() > 1)
    {
        sizeY = vector.at(1);
    }
    if (vector.size() > 2)
    {
        sizeZ = vector.at(2);
    }
}

void DimensionVector::setSizeX(const size_t sizeX)
{
    this->sizeX = sizeX;
}

void DimensionVector::setSizeY(const size_t sizeY)
{
    this->sizeY = sizeY;
}

void DimensionVector::setSizeZ(const size_t sizeZ)
{
    this->sizeZ = sizeZ;
}

void DimensionVector::multiply(const DimensionVector& factor)
{
    sizeX *= factor.sizeX;
    sizeY *= factor.sizeY;
    sizeZ *= factor.sizeZ;
}

void DimensionVector::divide(const DimensionVector& divisor)
{
    sizeX /= divisor.sizeX;
    sizeY /= divisor.sizeY;
    sizeZ /= divisor.sizeZ;
}

void DimensionVector::modifyByValue(const size_t value, const ThreadModifierAction& modifierAction, const Dimension modifierDimension)
{
    switch (modifierAction)
    {
    case ThreadModifierAction::Add:
        addValue(value, modifierDimension);
        break;
    case ThreadModifierAction::Subtract:
        subtractValue(value, modifierDimension);
        break;
    case ThreadModifierAction::Multiply:
        multiplyByValue(value, modifierDimension);
        break;
    case ThreadModifierAction::Divide:
        divideByValue(value, modifierDimension);
        break;
    default:
        throw std::runtime_error("Unknown thread modifier action");
    }
}

size_t DimensionVector::getSizeX() const
{
    return sizeX;
}

size_t DimensionVector::getSizeY() const
{
    return sizeY;
}

size_t DimensionVector::getSizeZ() const
{
    return sizeZ;
}

size_t DimensionVector::getTotalSize() const
{
    return sizeX * sizeY * sizeZ;
}

std::vector<size_t> DimensionVector::getVector() const
{
    std::vector<size_t> vector;
    vector.push_back(sizeX);
    vector.push_back(sizeY);
    vector.push_back(sizeZ);
    return vector;
}

bool DimensionVector::operator==(const DimensionVector& other) const
{
    return sizeX == other.sizeX && sizeY == other.sizeY && sizeZ == other.sizeZ;
}

bool DimensionVector::operator!=(const DimensionVector& other) const
{
    return !(*this == other);
}

void DimensionVector::addValue(const size_t value, const Dimension modifierDimension)
{
    if (modifierDimension == Dimension::X)
    {
        sizeX += value;
    }
    else if (modifierDimension == Dimension::Y)
    {
        sizeY += value;
    }
    else
    {
        sizeZ += value;
    }
}

void DimensionVector::subtractValue(const size_t value, const Dimension modifierDimension)
{
    if (modifierDimension == Dimension::X)
    {
        sizeX -= value;
    }
    else if (modifierDimension == Dimension::Y)
    {
        sizeY -= value;
    }
    else
    {
        sizeZ -= value;
    }
}

void DimensionVector::multiplyByValue(const size_t value, const Dimension modifierDimension)
{
    if (modifierDimension == Dimension::X)
    {
        sizeX *= value;
    }
    else if (modifierDimension == Dimension::Y)
    {
        sizeY *= value;
    }
    else
    {
        sizeZ *= value;
    }
}

void DimensionVector::divideByValue(const size_t value, const Dimension modifierDimension)
{
    if (modifierDimension == Dimension::X)
    {
        sizeX /= value;
    }
    else if (modifierDimension == Dimension::Y)
    {
        sizeY /= value;
    }
    else
    {
        sizeZ /= value;
    }
}

std::ostream& operator<<(std::ostream& outputTarget, const DimensionVector& dimensionVector)
{
    outputTarget << dimensionVector.sizeX << ", " << dimensionVector.sizeY << ", " << dimensionVector.sizeZ;
    return outputTarget;
}

} // namespace ktt
