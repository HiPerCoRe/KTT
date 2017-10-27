#pragma once

#include <iostream>
#include <vector>
#include "ktt_platform.h"
#include "enum/dimension.h"
#include "enum/thread_modifier_action.h"

namespace ktt
{

class KTT_API DimensionVector
{
public:
    DimensionVector();
    explicit DimensionVector(const size_t sizeX);
    explicit DimensionVector(const size_t sizeX, const size_t sizeY);
    explicit DimensionVector(const size_t sizeX, const size_t sizeY, const size_t sizeZ);
    explicit DimensionVector(const std::vector<size_t>& vector);

    void setSizeX(const size_t sizeX);
    void setSizeY(const size_t sizeY);
    void setSizeZ(const size_t sizeZ);
    void multiply(const DimensionVector& factor);
    void divide(const DimensionVector& divisor);
    void modifyByValue(const size_t value, const ThreadModifierAction& modifierAction, const Dimension modifierDimension);

    size_t getSizeX() const;
    size_t getSizeY() const;
    size_t getSizeZ() const;
    size_t getTotalSize() const;
    std::vector<size_t> getVector() const;

    bool operator==(const DimensionVector& other) const;
    bool operator!=(const DimensionVector& other) const;
    KTT_API friend std::ostream& operator<<(std::ostream&, const DimensionVector&);

private:
    size_t sizeX;
    size_t sizeY;
    size_t sizeZ;

    void addValue(const size_t value, const Dimension modifierDimension);
    void subtractValue(const size_t value, const Dimension modifierDimension);
    void multiplyByValue(const size_t value, const Dimension modifierDimension);
    void divideByValue(const size_t value, const Dimension modifierDimension);
};

KTT_API std::ostream& operator<<(std::ostream& outputTarget, const DimensionVector& dimensionVector);

} // namespace ktt
