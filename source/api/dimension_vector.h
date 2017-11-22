/** @file dimension_vector.h
  * @brief Functionality related to specifying thread sizes of a kernel.
  */
#pragma once

#include <iostream>
#include <vector>
#include "ktt_platform.h"
#include "enum/dimension.h"
#include "enum/thread_modifier_action.h"

namespace ktt
{

/** @class DimensionVector
  * @brief Class which holds information about either global or local thread size of a single kernel.
  */
class KTT_API DimensionVector
{
public:
    /** @fn DimensionVector()
      * @brief Default constructor, creates dimension vector with thread sizes in all dimensions set to 1.
      */
    DimensionVector();

    /** @fn explicit DimensionVector(const size_t sizeX)
      * @brief Constructor which creates dimension vector with specified thread size in dimension x and thread sizes in other dimensions set to 1.
      * @param sizeX Thread size in dimension x.
      */
    explicit DimensionVector(const size_t sizeX);

    /** @fn explicit DimensionVector(const size_t sizeX, const size_t sizeY)
      * @brief Constructor which creates dimension vector with specified thread sizes in dimensions x and y and thread size in dimension z set to 1.
      * @param sizeX Thread size in dimension x.
      * @param sizeY Thread size in dimension y.
      */
    explicit DimensionVector(const size_t sizeX, const size_t sizeY);

    /** @fn explicit DimensionVector(const size_t sizeX, const size_t sizeY, const size_t sizeZ)
      * @brief Constructor which creates dimension vector with specified thread sizes in all dimensions.
      * @param sizeX Thread size in dimension x.
      * @param sizeY Thread size in dimension y.
      * @param sizeZ Thread size in dimension z.
      */
    explicit DimensionVector(const size_t sizeX, const size_t sizeY, const size_t sizeZ);

    /** @fn explicit DimensionVector(const std::vector<size_t>& vector)
      * @brief Constructor which creates dimension vector with thread sizes based on up to first three elements of provided vector. If size of vector
      * is less than 3, remaining thread sizes are set to 1.
      * @param vector Source vector for dimension vector thread sizes.
      */
    explicit DimensionVector(const std::vector<size_t>& vector);

    /** @fn void setSizeX(const size_t sizeX)
      * @brief Setter for thread size in dimension x.
      * @param sizeX Thread size in dimension x.
      */
    void setSizeX(const size_t sizeX);

    /** @fn void setSizeY(const size_t sizeY)
      * @brief Setter for thread size in dimension y.
      * @param sizeY Thread size in dimension y.
      */
    void setSizeY(const size_t sizeY);

    /** @fn void setSizeZ(const size_t sizeZ)
      * @brief Setter for thread size in dimension z.
      * @param sizeZ Thread size in dimension z.
      */
    void setSizeZ(const size_t sizeZ);

    /** @fn void multiply(const DimensionVector& factor)
      * @brief Multiplies thread sizes by values provided by specified dimension vector.
      * @param factor Source of values for thread size multiplication.
      */
    void multiply(const DimensionVector& factor);

    /** @fn void divide(const DimensionVector& divisor)
      * @brief Divides thread sizes by values provided by specified dimension vector.
      * @param divisor Source of values for thread size division.
      */
    void divide(const DimensionVector& divisor);

    /** @fn void modifyByValue(const size_t value, const ThreadModifierAction& modifierAction, const Dimension modifierDimension)
      * @brief Modifies thread size in single dimension based on provided value and action.
      * @param value Value which will modifies thread size in single dimension based on specified action.
      * @param modifierAction Specifies which operation should be performed with thread size and specified value.
      * @param modifierDimension Specifies which dimension will be affected by the action.
      */
    void modifyByValue(const size_t value, const ThreadModifierAction& modifierAction, const Dimension modifierDimension);

    /** @fn size_t getSizeX() const
      * @brief Getter for thread size in dimension x.
      * @return Thread size in dimension x.
      */
    size_t getSizeX() const;

    /** @fn size_t getSizeY() const
      * @brief Getter for thread size in dimension y.
      * @return Thread size in dimension y.
      */
    size_t getSizeY() const;

    /** @fn size_t getSizeZ() const
      * @brief Getter for thread size in dimension z.
      * @return Thread size in dimension z.
      */
    size_t getSizeZ() const;

    /** @fn size_t getTotalSize() const
      * @brief Getter for total thread size. Total thread size is calculated by multiplying thread sizes in each dimension.
      * @return Total thread size.
      */
    size_t getTotalSize() const;

    /** @fn std::vector<size_t> getVector() const
      * @brief Converts dimension vector to STL vector. Resulting vector will always contain 3 elements.
      * @return Converted STL vector.
      */
    std::vector<size_t> getVector() const;

    /** @fn bool operator==(const DimensionVector& other) const
      * @brief Comparison operator for dimension vector. Compares thread sizes in all 3 dimensions.
      * @return True if dimension vectors are equal. False otherwise.
      */
    bool operator==(const DimensionVector& other) const;

    /** @fn bool operator!=(const DimensionVector& other) const
      * @brief Comparison operator for dimension vector. Compares thread sizes in all 3 dimensions.
      * @return True if dimension vectors are not equal. False otherwise.
      */
    bool operator!=(const DimensionVector& other) const;

private:
    size_t sizeX;
    size_t sizeY;
    size_t sizeZ;

    void addValue(const size_t value, const Dimension modifierDimension);
    void subtractValue(const size_t value, const Dimension modifierDimension);
    void multiplyByValue(const size_t value, const Dimension modifierDimension);
    void divideByValue(const size_t value, const Dimension modifierDimension);
};

/** @fn std::ostream& operator<<(std::ostream& outputTarget, const DimensionVector& dimensionVector)
  * @brief Output operator for dimension vector class.
  * @param outputTarget Location where information about dimension vector is printed.
  * @param dimensionVector Dimension vector object that is printed.
  * @return Output target to support chain of output operations.
  */
KTT_API std::ostream& operator<<(std::ostream& outputTarget, const DimensionVector& dimensionVector);

} // namespace ktt
